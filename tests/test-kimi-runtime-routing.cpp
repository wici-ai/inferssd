// Launch-only K3 B/C snapshots plus actual small CPU MoE graph execution.
// Context fixtures are graphless; numeric fixtures execute the production MoE
// frontend with the frozen configurations. Full-model KV/KDA is not claimed.
#include "llama-context.h"
#include "llama-model.h"
#include "llama-cpu-sink.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"
#if defined(__linux__)
#include "linux-test-resources.h"
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <fstream>
#include <unistd.h>

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

struct env_guard {
    std::string name, value;
    bool present;
    explicit env_guard(const char * name) : name(name), present(std::getenv(name) != nullptr) {
        if (present) { value = std::getenv(name); }
        unsetenv(name);
    }
    ~env_guard() { if (present) { setenv(name.c_str(), value.c_str(), 1); } else { unsetenv(name.c_str()); } }
};

static llama_context_params context_params() {
    auto p = llama_context_default_params();
    p.n_ctx = 128; p.n_batch = 8; p.n_ubatch = 8; p.n_seq_max = 1;
    p.n_threads = p.n_threads_batch = 1;
    return p;
}

static void set_model_metadata(llama_model & model, llm_arch arch) {
    model.arch = arch;
    auto & hp = model.hparams;
    hp.vocab_only = true;
    hp.n_expert = hp.n_expert_used = 4;
    hp.n_layer = 3; hp.n_ctx_train = 128; hp.n_embd = 8;
    hp.n_head_arr.fill(1); hp.n_head_kv_arr.fill(1);
    hp.n_embd_head_k_full = hp.n_embd_head_v_full = 8;
    hp.expert_weights_norm = true;
}

struct moe_values {
    std::vector<int32_t> ids;
    std::vector<float> weights, output;
};

class moe_fixture {
    ggml_backend_t cpu;
    llm_graph_result result{2048};
    llama_adapter_loras loras;
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer{nullptr, ggml_backend_buffer_free};
    ggml_tensor * input = nullptr, * logits = nullptr, * up = nullptr, * gate = nullptr, * down = nullptr;
    ggml_tensor * ids = nullptr, * weights = nullptr, * output = nullptr;

public:
    int builds = 0, reused = 0;
    explicit moe_fixture(ggml_backend_t cpu) : cpu(cpu) {}

    moe_values run(const llama_k3_decode_expert_policy & policy, int tokens = 1, int layer = 1,
            const llama_k3_decode_expert_table & table = {}) {
        llm_graph_params params = {};
        params.arch = LLM_ARCH_KIMI_K3;
        params.hparams.n_embd = 8;
        params.hparams.n_layer = 3;
        params.hparams.n_expert = params.hparams.n_expert_used = 4;
        params.hparams.n_head_arr.fill(1); params.hparams.n_head_kv_arr.fill(1);
        params.hparams.n_embd_head_k_full = params.hparams.n_embd_head_v_full = 8;
        params.hparams.situ_beta = 4; params.hparams.situ_linear_beta = 25;
        params.cparams.n_ctx = params.cparams.n_ctx_seq = 128;
        params.cparams.n_batch = params.cparams.n_ubatch = 8;
        params.cparams.n_seq_max = 1;
        params.ubatch.n_tokens = params.ubatch.n_seq_tokens = tokens;
        params.ubatch.n_seqs = params.ubatch.n_seqs_unq = 1;
        params.ubatch.b_equal_seqs = 1;
        params.ubatch.data = std::make_shared<llama_ubatch::data_t>();
        params.ubatch.data->seq_id_unq.push_back(0);
        params.ubatch.seq_id_unq = params.ubatch.data->seq_id_unq.data();
        params.gtype = LLM_GRAPH_TYPE_DEFAULT;
        params.backend_cpu = cpu;
        params.loras = &loras;
        params.n_outputs = 1;
        params.k3_expert_policy = policy;
        require(policy.valid(), "fixture policy rejected");
        require(llama_k3_expert_table_canonicalize(table.data(), table.size(), 4, 3, params.k3_expert_table),
            "fixture table rejected");
        params.res = &result;
        params.cb = [this](const llama_ubatch &, ggml_tensor * tensor, const char * name, int) {
            if (std::string(name) == "ffn_moe_topk" || std::string(name) == "ffn_moe_mass_ids") { ids = tensor; }
            if (std::string(name) == "ffn_moe_weights_norm" || std::string(name) == "ffn_moe_mass_weights_norm") {
                weights = tensor;
            }
        };
        // A fixture instance always uses one layer; production graph builds all
        // layers. Each alternate layer case below gets its own cold fixture.
        if (result.can_reuse(params)) {
            ++reused;
        } else {
            buffer.reset(); result.reset(); ++builds;
            llm_graph_context context(params);
            auto * ctx = result.get_ctx();
            input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, tokens);
            logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, tokens);
            up = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 8, 4);
            gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 8, 4);
            down = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 8, 4);
            const auto args = llama_k3_expert_graph_args(params.k3_expert_policy, params.k3_expert_table, tokens, layer);
            output = context.build_moe_ffn(input, nullptr, up, gate, down, nullptr,
                4, 4, LLM_FFN_SITU, true, 1.0f, LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
                layer, logits, nullptr, nullptr, nullptr, nullptr,
                args.hard_top_k, args.mass_threshold, args.mass_min);
            ggml_build_forward_expand(result.get_gf(), output);
            require(ids && weights, "MoE graph did not expose IDs/weights");
            buffer.reset(ggml_backend_alloc_ctx_tensors(ctx, cpu));
            require(bool(buffer), "CPU graph allocation failed");
            std::vector<float> v(8 * tokens, 0.5f);
            ggml_backend_tensor_set(input, v.data(), 0, v.size() * sizeof(float));
            v.resize(4 * tokens);
            const float probabilities[] = {0.5f, 0.25f, 0.15f, 0.10f};
            for (size_t i = 0; i < v.size(); ++i) { v[i] = std::log(probabilities[i % 4] / (1 - probabilities[i % 4])); }
            ggml_backend_tensor_set(logits, v.data(), 0, v.size() * sizeof(float));
            v.resize(8 * 8 * 4);
            for (size_t i = 0; i < v.size(); ++i) { v[i] = 0.02f * (1 + i / 64); }
            for (auto * tensor : {up, gate, down}) { ggml_backend_tensor_set(tensor, v.data(), 0, v.size() * sizeof(float)); }
        }
        require(ggml_backend_graph_compute(cpu, result.get_gf()) == GGML_STATUS_SUCCESS, "CPU MoE compute failed");
        moe_values value;
        value.ids.resize(ggml_nelements(ids)); value.weights.resize(ggml_nelements(weights));
        value.output.resize(ggml_nelements(output));
        ggml_backend_tensor_get(ids, value.ids.data(), 0, value.ids.size() * sizeof(int32_t));
        ggml_backend_tensor_get(weights, value.weights.data(), 0, value.weights.size() * sizeof(float));
        ggml_backend_tensor_get(output, value.output.data(), 0, value.output.size() * sizeof(float));
        return value;
    }
};

static void equal_values(const moe_values & a, const moe_values & b) {
    require(a.ids == b.ids, "cold/switch expert IDs differ");
    require(a.weights.size() == b.weights.size() && a.output.size() == b.output.size(), "cold/switch shapes differ");
    for (size_t i = 0; i < a.weights.size(); ++i) {
        require(std::abs(a.weights[i] - b.weights[i]) < 1e-6f, "cold/switch weights differ");
    }
    for (size_t i = 0; i < a.output.size(); ++i) {
        require(std::isfinite(a.output[i]) && std::abs(a.output[i] - b.output[i]) < 1e-6f, "cold/switch MoE outputs differ");
    }
}

static void expect_routes(const moe_values & value, int kept, const std::vector<float> & expected) {
    require(value.ids.size() == 4 && value.weights.size() == 4, "single-row route size mismatch");
    unsigned seen = 0;
    for (int i = 0; i < 4; ++i) {
        const int id = value.ids[i];
        // The full-width plain top-k graph preserves its native (heap) order;
        // truncation graphs sort the retained prefix. Validate the expert set
        // and each expert's weight here; cold/switch comparisons check order.
        if (id == -1) {
            require(std::abs(value.weights[i]) < 1e-5f, "dropped expert has nonzero weight");
            continue;
        }
        require(id >= 0 && id < kept && !(seen & (1u << id)), "incorrect retained expert set");
        seen |= 1u << id;
        require(std::abs(value.weights[i] - expected[id]) < 1e-5f, "incorrect normalized expert weight");
    }
    require(seen == (1u << kept) - 1, "missing retained expert");
}

template<typename F> static void rejects(F && f) {
    bool rejected = false;
    try { f(); } catch (const std::exception &) { rejected = true; }
    require(rejected, "invalid launch configuration accepted");
}

static void cache_launch_tests() {
    env_guard bytes("GGML_KIMI_CPU_SINK_CACHE_BYTES");
    env_guard quota("GGML_KIMI_CPU_SINK_CACHE_MAX_PER_LAYER");
    env_guard priority("GGML_KIMI_CPU_SINK_CACHE_LAYER_PRIORITY");
    // unset: the sink sizes an automatic cache at the first decode token, bounded per layer by the experts alone
    const llama_kimi_cpu_sink_runtime defaults(93, 1);
    require(defaults.cache_budget == std::numeric_limits<uint64_t>::max() && defaults.cache_max_per_layer == 896,
        "cache launch defaults changed");
    for (const char * value : {"0", "4000000000", "100000000000", "594000000000", "100000000001"}) {
        setenv(bytes.name.c_str(), value, 1);
        setenv(quota.name.c_str(), "896", 1);
        const llama_kimi_cpu_sink_runtime runtime(93, 1);
        require(runtime.cache_budget == std::stoull(value) && runtime.cache_max_per_layer == 896,
            "large or unaligned cache launch value lost");
    }
    for (const char * value : {"", "-1", "+1", " 1", "1 ", "1GB", "0x10", "18446744073709551616"}) {
        setenv(bytes.name.c_str(), value, 1);
        rejects([] { llama_kimi_cpu_sink_runtime runtime(93, 1); });
    }
    setenv(bytes.name.c_str(), "100000000000", 1);
    for (const char * value : {"0", "897", "-1", "+8", "8 ", "", "18446744073709551616"}) {
        setenv(quota.name.c_str(), value, 1);
        rejects([] { llama_kimi_cpu_sink_runtime runtime(93, 1); });
    }
    setenv(quota.name.c_str(), "1", 1);
    require(llama_kimi_cpu_sink_runtime(93, 1).cache_max_per_layer == 1, "minimum cache quota rejected");
    std::printf("KIMI_CACHE_LAUNCH_OK large_bytes=PASS unaligned_bytes=PASS quota_896=PASS rejects=15\n");
}

struct table_file {
    char path[64] = "/tmp/llama-k3-layer-table-XXXXXX";
    table_file() {
        const int fd = mkstemp(path);
        require(fd >= 0, "cannot create table fixture file");
        close(fd);
    }
    void write(const std::string & text) {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        file << text;
        require(bool(file), "cannot write table fixture file");
    }
    ~table_file() { std::remove(path); }
};

static const llama_k3_runtime_config & config(const llama_context & ctx) {
    const auto * value = llama_k3_runtime_config_get(&ctx);
    require(value != nullptr, "missing K3 runtime config");
    return *value;
}

static void table_parse_tests() {
    // The supplied v1 header contains a semicolon inside a # comment. It must
    // stay a comment through the physical newline, including inline comments.
    const auto commented = llama_k3_expert_table_parse(
        "# K3 per-layer routing table v1 (derived from the 09-09 skip-scan KL ranking; zero-based MoE layer index)\n"
        "4-5 .85 4 8;12 .70 2 4 # ignored ;13 .70 2 4\n"
        "36 .80 2 6;37 .80 2 6", 16, 93);
    require(commented.size() == 5 && llama_k3_expert_tables_equal(commented,
        llama_k3_expert_table_parse("4-5 .85 4 8;12 .70 2 4;36-37 .80 2 6", 16, 93)),
        "comment semicolon started a record or physical newline lost following records");
    const auto canonical = llama_k3_expert_table_parse("2 .95 1 3; 1 .7 1 3", 4, 3);
    require(canonical.size() == 2 && canonical[0].layer == 1 && canonical[1].layer == 2, "table not canonicalized");
    require(llama_k3_expert_tables_equal(canonical,
        llama_k3_expert_table_parse("# comment\n 1 0.70 1 3\n2 0.950 1 3\n", 4, 3)), "equivalent records differ");
    const auto range = llama_k3_expert_table_parse("0-2 .8 1 0", 4, 3);
    require(range.size() == 3 && range[2].layer == 2 && range[0].max_k == 0, "range/uncapped parse failed");
    for (const char * text : {
            "1 .7 1 3;1 .8 1 3", "0-1 .7 1 3;1-2 .8 1 3", "2-1 .7 1 3", "-1 .7 1 3", "3 .7 1 3",
            "1 nan 1 3", "1 inf 1 3", "1 0 1 3", "1 1.01 1 3", "1 .7 0 3", "1 .7 5 0", "1 .7 1 5",
            "1 .7 1 -1", "1 .7 4 3", "1 .7 1", "1 .7 1 3 extra", "1x .7 1 3", "1 .7 1.5 3" }) {
        rejects([&] { llama_k3_expert_table_parse(text, 4, 3); });
    }
    rejects([&] { llama_k3_expert_table_parse(std::string(65537, ' '), 4, 3); });
    rejects([&] { llama_k3_expert_table_parse("", 0, 0); });
    rejects([&] { llama_k3_expert_table_parse(std::string("1 .7 1 3\0garbage\n", 17), 4, 3); });
}

static void launch_reject_tests(llama_model & model) {
    for (const auto & invalid : std::vector<std::pair<const char *, const char *>>{
            {"TOKENS", "-1"}, {"TOKENS", "2147483648"}, {"TOKENS", "1.5"},
            {"START_TOKENS", "-1"}, {"START_TOKENS", "2147483648"}, {"START_TOKENS", "1.5"},
            {"START_TOKENS", "1x"}, {"START_TOKENS", "9223372036854775808"},
            {"CONTROL_TOKENS", "-1"}, {"CONTROL_TOKENS", "2147483648"}, {"CONTROL_TOKENS", "1.5"},
            {"CONTROL_TOKENS", "1x"}, {"CONTROL_TOKENS", "9223372036854775808"},
            {"MODE", "invalid"}, {"TOP_K", "0"}, {"TOP_K", "5"},
            {"MASS", "0"}, {"MASS", "1.01"}, {"MASS", "nan"}, {"MASS", "inf"},
            {"MIN_K", "0"}, {"MIN_K", "5"}, {"MAX_K", "-1"}, {"MAX_K", "5"}}) {
        const std::string name = std::string("LLAMA_K3_DECODE_EXPERT_GUARD_") + invalid.first;
        env_guard restore(name.c_str());
        setenv(name.c_str(), invalid.second, 1);
        rejects([&] { llama_context rejected(model, context_params()); });
    }
    {
        env_guard mode("LLAMA_K3_DECODE_EXPERT_GUARD_MODE");
        env_guard minimum("LLAMA_K3_DECODE_EXPERT_GUARD_MIN_K");
        env_guard maximum("LLAMA_K3_DECODE_EXPERT_GUARD_MAX_K");
        setenv(mode.name.c_str(), "mass", 1);
        setenv(minimum.name.c_str(), "4", 1);
        setenv(maximum.name.c_str(), "3", 1);
        rejects([&] { llama_context invalid_combination(model, context_params()); });
    }
}

// The prefill route of each ubatch: the CPU sink up to GGML_KIMI_CPU_PREFILL_MAX_TOKENS (or GGML_KIMI_CPU_PREFILL: 1
// every ubatch, 0 none; not both) and for any ubatch the GPU would not take, the ggml graph above.  A CPU-only build has no GPU
// to offload to, so every prefill ubatch of a sink context runs on the sink here.
static void prefill_route_tests() {
    const uint32_t all = std::numeric_limits<uint32_t>::max();
    require(!llama_cpu_sink_prefill_route(false, all, 1, false), "a context without the sink routed a ubatch to it");
    require(llama_cpu_sink_prefill_route(true, 0, 31, false), "a ubatch the GPU does not take left the sink");
    require(!llama_cpu_sink_prefill_route(true, 0, 32, true), "threshold 0 kept an offloaded ubatch on the sink");
    require(llama_cpu_sink_prefill_route(true, 500, 500, true) && !llama_cpu_sink_prefill_route(true, 500, 501, true),
        "the threshold is not the largest sink ubatch");
    require(llama_cpu_sink_prefill_route(true, all, all, true), "CPU_PREFILL=1 sent a ubatch to the graph");

    llama_model model(llama_model_default_params());
    set_model_metadata(model, LLM_ARCH_KIMI_K3);
    model.layers.resize(model.hparams.n_layer);
    env_guard sink("GGML_KIMI_CPU_SINK");
    env_guard prefill("GGML_KIMI_CPU_PREFILL");
    env_guard max_tokens("GGML_KIMI_CPU_PREFILL_MAX_TOKENS");
    env_guard actual_shape("GGML_KIMI_CPU_PREFILL_ACTUAL_SHAPE");
    auto threshold = [&](const char * sink_value, const char * prefill_value, const char * max_value) {
        for (auto * guard : { &sink, &prefill, &max_tokens }) {
            unsetenv(guard->name.c_str());
        }
        if (sink_value) { setenv(sink.name.c_str(), sink_value, 1); }
        if (prefill_value) { setenv(prefill.name.c_str(), prefill_value, 1); }
        if (max_value) { setenv(max_tokens.name.c_str(), max_value, 1); }
        llama_context ctx(model, context_params());
        for (uint32_t n : { 1u, 8u, 32u, 8192u }) {
            require(ctx.cpu_sink_prefill_route(n) == (sink_value != nullptr), "a prefill ubatch on the wrong route");
        }
        return ctx.cpu_prefill_max_tokens_get();
    };
    require(threshold(nullptr, "0", nullptr) == 0 && threshold("1", nullptr, nullptr) == 0, "default threshold");
    require(threshold("1", "1", nullptr) == all && threshold("1", "0", nullptr) == 0, "CPU_PREFILL spellings");
    require(threshold("1", nullptr, "500") == 500 && threshold("1", nullptr, "0") == 0 &&
        threshold("1", nullptr, "0500") == 500 && threshold("1", nullptr, "4294967295") == all, "MAX_TOKENS does not decide");
    for (const char * invalid : { "-1", "+5", " 5", "5x", "1.5", "abc", "4294967296" }) {
        rejects([&] { threshold("1", nullptr, invalid); });
    }
    rejects([&] { threshold("1", "1", "0"); });
    rejects([&] { threshold("1", "0", "500"); });
    {
        // the legacy GPU union workspace: refused when every ubatch takes the sink (it would serve none), kept below
        // a finite threshold, where it serves the graph route (K3's phase-workspace profiles with the sink on)
        env_guard phase0("GGML_MOE_PREFILL_PHASE0");
        setenv(phase0.name.c_str(), "1", 1);
        rejects([&] { threshold("1", "1", nullptr); });
        require(threshold("1", "0", nullptr) == 0 && threshold("1", nullptr, "500") == 500,
            "the workspace refused below a finite threshold");
        require(threshold(nullptr, "0", nullptr) == 0, "the workspace refused without the sink");
    }
    rejects([&] { threshold(nullptr, nullptr, "500"); });
    rejects([&] { threshold(nullptr, "1", nullptr); });
    rejects([&] { threshold("1", "2", nullptr); });
    // graphs sized to the request on either route; the phase transitions need the sink
    setenv(actual_shape.name.c_str(), "1", 1);
    require(threshold("1", "0", nullptr) == 0, "actual shape refused on the graph route");
    rejects([&] { threshold(nullptr, "0", nullptr); });
    unsetenv(actual_shape.name.c_str());
    std::printf("KIMI_PREFILL_ROUTE_OK rule=PASS threshold=PASS rejects=13 actual_shape_both_routes=PASS\n");
}

// The [moe_prefill_copy] line: the scheduler's counters over one prefill's graph-route passes, nothing for a prefill
// that ran only on the sink.
static void prefill_copy_tests() {
    llama_prefill_copy_meter meter;
    meter.begin({ 100, 1000, 10, 1 });
    require(meter.end({ 200, 9000, 50, 1 }).empty(), "a sink-only prefill printed a copy line");
    meter.begin({ 100, 1000, 10, 1 });
    meter.pass(4096);
    meter.pass(125);
    const std::string line = meter.end({ 37582, 379930085864, 34319475, 1 });
    require(line == "[moe_prefill_copy] tokens=4221 passes=2 expert_slices=37482 bytes=379930084864 read_us=34319465 "
        "read_gbs=11.07 fallbacks=0", ("the copy line: " + line).c_str());
    require(meter.end({ 37582, 379930085864, 34319475, 1 }).empty(), "a copy line printed twice");
    meter.begin({ 0, 0, 0, 0 });
    meter.pass(669);
    require(meter.end({ 3, 300, 0, 2 }) == "[moe_prefill_copy] tokens=669 passes=1 expert_slices=3 bytes=300 read_us=0 "
        "read_gbs=0.00 fallbacks=2", "the copy line without read time");
    std::printf("KIMI_PREFILL_COPY_OK sink_only=PASS delta=PASS once=PASS\n");
}

// The GPU's offload test for the route, asked of the devices of this build: with no GPU nothing is offloaded; a GPU
// offloads from GGML_OP_OFFLOAD_MIN_BATCH tokens (32 by default), and only when it supports every expert tensor's type.
static void offload_probe_tests() {
    const ggml_init_params params = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr ctx { ggml_init(params) };
    const ggml_tensor * up = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q4_K, 256, 512, 8);
    const ggml_tensor * down = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q5_K, 512, 256, 8);
    const ggml_tensor * unsupported = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_TQ2_0, 256, 512, 8);
    std::vector<ggml_backend_dev_t> gpus;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(dev);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            gpus.push_back(dev);
        }
    }
    auto offloads = [&](std::vector<const ggml_tensor *> experts, uint32_t n) {
        return llama_gpu_offloads_mul_mat_id(gpus, experts, 8, n);
    };
    require(!offloads({}, 4096), "no expert tensor was offloaded");
    if (gpus.empty()) {
        require(!offloads({ up, down }, 4096), "a build without a GPU offloaded");
        std::printf("KIMI_OFFLOAD_PROBE_OK gpus=0\n");
        return;
    }
    const char * floor_env = std::getenv("GGML_OP_OFFLOAD_MIN_BATCH");
    const uint32_t floor = floor_env ? uint32_t(std::atoi(floor_env)) : 32;
    require(floor > 1 && !offloads({ up, down }, floor - 1) && offloads({ up, down }, floor) && offloads({ up, down }, 4221),
        "the GPU's offload floor");
    require(!offloads({ up, unsupported }, 4096), "an expert type the GPU does not support was offloaded");
    std::printf("KIMI_OFFLOAD_PROBE_OK gpus=%zu floor=%u unsupported_type=PASS\n", gpus.size(), floor);
}

// The request-start reserve when the request's own graph does not fit: the largest graph size above the threshold,
// else one sink pass of the whole request, else the largest sink size.  The fixture's GPU offloads from 32 tokens,
// its graph fits up to graph_fit tokens and its sink up to sink_fit; reserve() sees the threshold the search holds.
static void prefill_search_tests() {
    const uint32_t all = std::numeric_limits<uint32_t>::max();
    struct outcome { uint32_t ubatch, max_sink_tokens; std::vector<uint32_t> reserves; };
    auto search = [&](uint32_t target, uint32_t max_sink_tokens, uint32_t graph_fit, uint32_t sink_fit) {
        outcome out = { 0, max_sink_tokens, {} };
        auto sink_route = [](uint32_t n, uint32_t max_tokens) { return llama_cpu_sink_prefill_route(true, max_tokens, n, n >= 32); };
        out.ubatch = llama_prefill_ubatch_search(target, 1, out.max_sink_tokens, sink_route, [&](uint32_t n) {
            out.reserves.push_back(n);
            return n <= (sink_route(n, out.max_sink_tokens) ? sink_fit : graph_fit);
        });
        return out;
    };
    require(search(4221, 0, all, all).ubatch == 4221 && search(4221, 0, all, all).reserves.size() == 1,
        "a request whose graph fits was searched");
    // the graph fits below the request: the largest graph size, the threshold kept
    auto r = search(4221, 0, 2048, all);
    require(r.ubatch == 2048 && r.max_sink_tokens == 0, "the graph fallback is not the largest graph size");
    r = search(4221, 500, 1000, all);
    require(r.ubatch == 768 && r.max_sink_tokens == 500, "a graph size at or below the threshold was taken");
    // no graph size fits: one sink pass of the request (threshold raised), else the largest sink size that fits
    for (uint32_t max_sink_tokens : { 0u, 200u, 500u }) {
        r = search(4221, max_sink_tokens, max_sink_tokens, all);
        require(r.ubatch == 4221 && r.max_sink_tokens == 4221, "the sink did not take the whole request");
        for (uint32_t n : r.reserves) {
            require(n == 4221 || n > max_sink_tokens, "the graph search reserved a sink size");
        }
    }
    r = search(4221, 0, 0, 1000);
    require(r.ubatch == 768 && r.max_sink_tokens == 4221, "the sink fallback is not the largest sink size");
    // a request below the offload floor's double: no graph size, a sink pass of the request
    r = search(40, 0, 0, all);
    require(r.ubatch == 40 && r.max_sink_tokens == 40 && r.reserves == std::vector<uint32_t>({ 40, 40 }),
        "a 40-token request took a sink size below it");
    // every ubatch on the sink (CPU_PREFILL=1): the search of the sink sizes as before, then one token
    r = search(8192, all, 0, 3000);
    require(r.ubatch == 2816 && r.max_sink_tokens == all &&
        r.reserves == std::vector<uint32_t>({ 8192, 4096, 2048, 3072, 2560, 2816 }), "the sink search changed");
    r = search(300, all, 0, 1);
    require(r.ubatch == 1 && r.reserves == std::vector<uint32_t>({ 300, 150, 1 }), "the one-token fallback changed");
    r = search(300, all, 0, 0);
    require(r.ubatch == 0, "a size was selected that does not fit");
    std::printf("KIMI_PREFILL_SEARCH_OK graph=PASS sink_whole_request=PASS sink=PASS legacy=PASS\n");
}

static void graph_identity_tests(const llama_k3_runtime_config & cfg) {
    llm_graph_params key{};
    key.arch = LLM_ARCH_KIMI_K3;
    key.k3_expert_policy = cfg.primary;
    key.k3_expert_table = cfg.table;
    require(key.allow_reuse(key), "identical empty graph identity rejected");
    for (int field = 0; field < 8; ++field) {
        auto changed = key;
        auto & p = changed.k3_expert_policy;
        switch (field) {
            case 0: p.mode = llama_k3_decode_expert_mode::disabled; break;
            case 1: --p.top_k; break;
            case 2: p.mass += .01f; break;
            case 3: ++p.min_k; break;
            case 4: ++p.max_k; break;
            case 5: --p.layer_start; break;
            case 6: --p.layer_end; break;
            case 7: p.small_batch = !p.small_batch; break;
        }
        require(!key.allow_reuse(changed), "policy graph identity field omitted");
    }
    for (int field = 0; field < 4; ++field) {
        auto changed = key;
        auto & row = changed.k3_expert_table[0];
        switch (field) {
            case 0: --row.layer; break;
            case 1: row.mass += .01f; break;
            case 2: ++row.min_k; break;
            case 3: --row.max_k; break;
        }
        require(!key.allow_reuse(changed), "table graph identity field omitted");
    }
    auto cleared = key; cleared.k3_expert_table.clear();
    require(!key.allow_reuse(cleared), "table clear reused primary graph");
}

int main() {
    try {
#if defined(__linux__)
        const auto sink_threads = llama_kimi_cpu_sink_runtime::unsigned_env("GGML_KIMI_CPU_SINK_THREADS", 16, 256);
        test_resources::cpu_threads(int(sink_threads) - 1, 1);
        test_resources::io_uring();
#endif
        cache_launch_tests();
        prefill_route_tests();
        prefill_search_tests();
        offload_probe_tests();
        prefill_copy_tests();
        std::vector<std::unique_ptr<env_guard>> environment;
        for (const char * name : {
                "LLAMA_K3_DECODE_EXPERT_MODE", "LLAMA_K3_DECODE_EXPERT_TOP_K", "LLAMA_K3_DECODE_EXPERT_MASS",
                "LLAMA_K3_DECODE_EXPERT_MIN_K", "LLAMA_K3_DECODE_EXPERT_MAX_K", "LLAMA_K3_DECODE_EXPERT_LAYER_START",
                "LLAMA_K3_DECODE_EXPERT_LAYER_END", "LLAMA_K3_DECODE_EXPERT_SMALL_BATCH", "LLAMA_K3_DECODE_EXPERT_LAYER_TABLE",
                "LLAMA_K3_DECODE_EXPERT_LAYER_TABLE_FILE", "LLAMA_K3_DECODE_EXPERT_GUARD_TOKENS", "LLAMA_K3_DECODE_EXPERT_GUARD_MODE",
                "LLAMA_K3_DECODE_EXPERT_GUARD_START_TOKENS",
                "LLAMA_K3_DECODE_EXPERT_GUARD_CONTROL_TOKENS",
                "LLAMA_K3_DECODE_EXPERT_GUARD_TOP_K", "LLAMA_K3_DECODE_EXPERT_GUARD_MASS", "LLAMA_K3_DECODE_EXPERT_GUARD_MIN_K",
                "LLAMA_K3_DECODE_EXPERT_GUARD_MAX_K", "LLAMA_K3_DECODE_SKIP_MOE_LAYERS_FILE",
                "GGML_KIMI_CPU_SINK", "GGML_KIMI_CPU_PREFILL", "GGML_KIMI_CPU_PREFILL_MAX_TOKENS",
                "GGML_KIMI_CPU_PREFILL_ACTUAL_SHAPE", "GGML_MOE_PREFILL_PHASE_SIZED_WORKSPACE",
                "GGML_MOE_PREFILL_DECODE_CACHE_HANDOVER", "GGML_KIMI_DENSE_PREFETCH_LAYERS", "GGML_KIMI_KDA_GPU_LAYERS",
                "GGML_MOE_PREFILL_PHASE0", "GGML_MOE_PREFILL_PHASE1", "GGML_KIMI_PIPELINE_V2_ACTIVE",
                "GGML_MOE_NEXT_GATE_SHADOW_OUT", "GGML_MOE_ROUTE_SCORE_TRACE_OUT" }) {
            environment.emplace_back(new env_guard(name));
        }
        llama_model model(llama_model_default_params());
        set_model_metadata(model, LLM_ARCH_KIMI_K3);
        llama_context disabled(model, context_params());
        require(config(disabled).guard_tokens == 0 && config(disabled).guard_start_tokens == 0 &&
            config(disabled).guard_control_tokens == 0 && config(disabled).table.empty() &&
            config(disabled).primary.mode == llama_k3_decode_expert_mode::disabled &&
            config(disabled).guard.mode == llama_k3_decode_expert_mode::hard && config(disabled).guard.top_k == 4,
            "default-off runtime config changed");
        require(!llama_k3_runtime_guard_set(&disabled, true) && llama_k3_runtime_guard_set(&disabled, false),
            "disabled guard accepted activation");
        table_parse_tests();

        setenv("LLAMA_K3_DECODE_EXPERT_MODE", "hard", 1);
        setenv("LLAMA_K3_DECODE_EXPERT_TOP_K", "4", 1);
        setenv("LLAMA_K3_DECODE_EXPERT_LAYER_START", "1", 1);
        setenv("LLAMA_K3_DECODE_EXPERT_LAYER_END", "2", 1);
        setenv("LLAMA_K3_DECODE_EXPERT_GUARD_TOKENS", "2", 1);
        setenv("LLAMA_K3_DECODE_EXPERT_GUARD_START_TOKENS", "3", 1);
        setenv("LLAMA_K3_DECODE_EXPERT_GUARD_CONTROL_TOKENS", "2", 1);
        setenv("LLAMA_K3_DECODE_EXPERT_LAYER_TABLE", "2 .95 1 3;1 .7 1 3", 1);
        llama_context primary(model, context_params());
        const auto * frozen = &config(primary);
        const auto snapshot = *frozen;
        require(frozen->primary.top_k == 4 && frozen->primary.layer_start == 1 && frozen->primary.layer_end == 2 &&
            !frozen->primary.small_batch && frozen->guard_tokens == 2 && frozen->guard_start_tokens == 3 &&
            frozen->guard_control_tokens == 2 && frozen->table.size() == 2 &&
            frozen->guard.layer_start == 1 && frozen->guard.layer_end == 2, "launch snapshot lost env semantics");
        setenv("LLAMA_K3_DECODE_EXPERT_GUARD_TOKENS", "0", 1);
        setenv("LLAMA_K3_DECODE_EXPERT_GUARD_START_TOKENS", "1", 1);
        setenv("LLAMA_K3_DECODE_EXPERT_GUARD_CONTROL_TOKENS", "0", 1);
        llama_context start_only(model, context_params());
        require(config(start_only).guard_tokens == 0 && config(start_only).guard_start_tokens == 1 &&
            config(start_only).guard_control_tokens == 0,
            "start-only context did not capture independent budgets");
        setenv("LLAMA_K3_DECODE_EXPERT_GUARD_START_TOKENS", "2147483647", 1);
        llama_context maximum_start(model, context_params());
        require(config(maximum_start).guard_start_tokens == 2147483647u &&
            llama_k3_runtime_guard_set(&maximum_start, true) && llama_k3_runtime_guard_set(&maximum_start, false),
            "valid maximum start budget rejected");
        setenv("LLAMA_K3_DECODE_EXPERT_GUARD_START_TOKENS", "0", 1);
        llama_context explicit_off(model, context_params());
        require(config(explicit_off).guard_start_tokens == 0 && config(explicit_off).guard_control_tokens == 0 &&
            !llama_k3_runtime_guard_set(&explicit_off, true), "explicit zero budgets accepted guard activation");
        setenv("LLAMA_K3_DECODE_EXPERT_GUARD_CONTROL_TOKENS", "1", 1);
        llama_context control_only(model, context_params());
        require(config(control_only).guard_tokens == 0 && config(control_only).guard_start_tokens == 0 &&
            config(control_only).guard_control_tokens == 1, "control-only context lost independent budget");
        setenv("LLAMA_K3_DECODE_EXPERT_GUARD_CONTROL_TOKENS", "2147483647", 1);
        llama_context maximum_control(model, context_params());
        require(config(maximum_control).guard_control_tokens == 2147483647u &&
            llama_k3_runtime_guard_set(&maximum_control, true) && llama_k3_runtime_guard_set(&maximum_control, false),
            "valid maximum control budget rejected");
        setenv("LLAMA_K3_DECODE_EXPERT_TOP_K", "2", 1);
        setenv("LLAMA_K3_DECODE_EXPERT_GUARD_TOKENS", "8", 1);
        setenv("LLAMA_K3_DECODE_EXPERT_GUARD_START_TOKENS", "8", 1);
        setenv("LLAMA_K3_DECODE_EXPERT_GUARD_CONTROL_TOKENS", "8", 1);
        setenv("LLAMA_K3_DECODE_EXPERT_LAYER_TABLE", "1 .2 1 3", 1);
        llama_context later(model, context_params());
        require(config(later).primary.top_k == 2 && config(later).guard_tokens == 8 &&
            config(later).guard_start_tokens == 8 && config(later).guard_control_tokens == 8 && config(later).table[0].mass == .2f,
            "new context did not capture its own env");
        require(config(primary).primary == snapshot.primary && config(primary).guard_tokens == 2 &&
            config(primary).guard_start_tokens == 3 && config(primary).guard_control_tokens == 2 &&
            config(start_only).guard_start_tokens == 1 && config(start_only).guard_control_tokens == 0 &&
            config(control_only).guard_control_tokens == 1 &&
            llama_k3_expert_tables_equal(config(primary).table, snapshot.table), "env mutation changed existing config");
        require(config(disabled).table.empty() && config(disabled).guard_tokens == 0 &&
            config(disabled).guard_start_tokens == 0 && config(disabled).guard_control_tokens == 0 &&
            !llama_k3_runtime_guard_set(&disabled, true),
            "runtime leaked across contexts");
        launch_reject_tests(model);

        unsetenv("LLAMA_K3_DECODE_EXPERT_LAYER_TABLE");
        table_file file;
        file.write("1 .7 1 3\n2 .95 1 3\n");
        setenv("LLAMA_K3_DECODE_EXPERT_LAYER_TABLE_FILE", file.path, 1);
        llama_context file_context(model, context_params());
        require(llama_k3_expert_tables_equal(config(file_context).table, snapshot.table), "file snapshot differs from inline table");
        file.write("1 .2 1 3\n");
        require(llama_k3_expert_tables_equal(config(file_context).table, snapshot.table), "file edit changed frozen table");
        llama_context file_later(model, context_params());
        require(config(file_later).table.size() == 1 && config(file_later).table[0].mass == .2f, "new context did not read file edit");
        file.write(std::string("1 .7 1 3\0garbage\n", 17));
        rejects([&] { llama_context invalid_file(model, context_params()); });
        setenv("LLAMA_K3_DECODE_EXPERT_LAYER_TABLE", "1 .7 1 3", 1);
        rejects([&] { llama_context conflicting_sources(model, context_params()); });
        unsetenv("LLAMA_K3_DECODE_EXPERT_LAYER_TABLE");
        setenv("LLAMA_K3_DECODE_EXPERT_LAYER_TABLE_FILE", "/nonexistent/k3-layer-table-fixture", 1);
        rejects([&] { llama_context missing_file(model, context_params()); });
        unsetenv("LLAMA_K3_DECODE_EXPERT_LAYER_TABLE_FILE");

        // Legacy OFF ignores raw START/END, so sparse C rows select the scope.
        setenv("LLAMA_K3_DECODE_EXPERT_MODE", "off", 1);
        setenv("LLAMA_K3_DECODE_EXPERT_LAYER_TABLE", "0 .7 1 3", 1);
        llama_context off_launch(model, context_params());
        require(config(off_launch).primary.layer_start == 0 && config(off_launch).primary.layer_end == 2 &&
            llama_k3_expert_graph_args(config(off_launch).primary, config(off_launch).table, 1, 0).mass_threshold == .7f,
            "C changed legacy off-mode layer-window semantics");
        model.hparams.expert_weights_norm = false;
        rejects([&] { llama_context unnormalized_table(model, context_params()); });
        unsetenv("LLAMA_K3_DECODE_EXPERT_LAYER_TABLE");
        setenv("LLAMA_K3_DECODE_EXPERT_GUARD_MODE", "mass", 1);
        setenv("LLAMA_K3_DECODE_EXPERT_GUARD_MASS", ".8", 1);
        rejects([&] { llama_context unnormalized_guard(model, context_params()); });
        model.hparams.expert_weights_norm = true;
        setenv("LLAMA_K3_DECODE_EXPERT_GUARD_MIN_K", "1", 1);
        setenv("LLAMA_K3_DECODE_EXPERT_GUARD_MAX_K", "3", 1);
        llama_context mass_guard(model, context_params());
        require(config(mass_guard).guard.mode == llama_k3_decode_expert_mode::mass && config(mass_guard).guard.mass == .8f &&
            config(mass_guard).guard.min_k == 1 && config(mass_guard).guard.max_k == 3, "guard mass launch lost fields");

        llama_model ordinary(llama_model_default_params());
        set_model_metadata(ordinary, LLM_ARCH_LLAMA);
        setenv("LLAMA_K3_DECODE_EXPERT_GUARD_CONTROL_TOKENS", "invalid-unused-model-env", 1);
        llama_context other(ordinary, context_params());
        llama_model tokenizer(llama_model_default_params());
        set_model_metadata(tokenizer, LLM_ARCH_KIMI_K3);
        tokenizer.hparams.n_expert_used = tokenizer.hparams.n_expert = tokenizer.hparams.n_layer = 0;
        setenv("LLAMA_K3_DECODE_EXPERT_MODE", "invalid-unused-tokenizer-env", 1);
        setenv("LLAMA_K3_DECODE_EXPERT_GUARD_MODE", "invalid-unused-tokenizer-env", 1);
        llama_context graphless(tokenizer, context_params());
        for (auto * ctx : {static_cast<llama_context *>(nullptr), &other, &graphless}) {
            require(llama_k3_runtime_config_get(ctx) == nullptr && !llama_k3_runtime_guard_set(ctx, true) &&
                !llama_k3_runtime_guard_set(ctx, false), "unavailable context accepted runtime controls");
        }

        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(), ggml_backend_free);
        require(bool(cpu), "CPU backend initialization failed");
        ggml_backend_cpu_set_n_threads(cpu.get(), 1);
        moe_fixture switched(cpu.get()), cold_primary(cpu.get()), cold_guard(cpu.get());
        auto pv = switched.run(frozen->primary, 1, 1, frozen->table);
        equal_values(pv, cold_primary.run(frozen->primary, 1, 1, frozen->table));
        expect_routes(pv, 2, {2.0f/3, 1.0f/3, 0, 0});
        auto reversed = frozen->table; std::reverse(reversed.begin(), reversed.end());
        equal_values(switched.run(frozen->primary, 1, 1, reversed), pv);
        require(switched.builds == 1 && switched.reused == 1, "same frozen form did not reuse graph");
        require(llama_k3_runtime_guard_set(&primary, true) && llama_k3_runtime_guard_set(&primary, true), "guard activation failed");
        auto gv = switched.run(frozen->guard);
        equal_values(gv, cold_guard.run(frozen->guard));
        expect_routes(gv, 4, {.5f, .25f, .15f, .1f});
        require(std::abs(pv.output[0] - gv.output[0]) > 1e-5f, "fixture insensitive to guard routing");
        require(llama_k3_runtime_guard_set(&primary, false) && llama_k3_runtime_guard_set(&primary, false), "primary restoration failed");
        equal_values(switched.run(frozen->primary, 1, 1, frozen->table), pv);
        require(switched.builds == 3, "primary-guard-primary failed to rebuild");
        require(&config(primary) == frozen && config(primary).primary == snapshot.primary &&
            config(primary).guard == snapshot.guard && config(primary).guard_start_tokens == snapshot.guard_start_tokens &&
            config(primary).guard_control_tokens == snapshot.guard_control_tokens &&
            llama_k3_expert_tables_equal(config(primary).table, snapshot.table),
            "bool switch mutated frozen config");
        graph_identity_tests(*frozen);

        // Start-only activation uses the same guard/empty-table graph as an
        // opener window, including primary restoration after the window ends.
        const auto * start_frozen = &config(start_only);
        moe_fixture start_switched(cpu.get());
        equal_values(start_switched.run(start_frozen->primary, 1, 1, start_frozen->table), pv);
        require(llama_k3_runtime_guard_set(&start_only, true) && llama_k3_runtime_guard_set(&start_only, true),
            "start-only guard activation rejected with zero opener budget");
        equal_values(start_switched.run(start_frozen->guard), gv);
        require(llama_k3_runtime_guard_set(&start_only, false), "start-only primary restoration failed");
        equal_values(start_switched.run(start_frozen->primary, 1, 1, start_frozen->table), pv);
        require(start_switched.builds == 3 && &config(start_only) == start_frozen &&
            start_frozen->guard_tokens == 0 && start_frozen->guard_start_tokens == 1 &&
            start_frozen->guard_control_tokens == 0 &&
            llama_k3_expert_tables_equal(start_frozen->table, snapshot.table),
            "start-only switch failed to rebuild or changed frozen configuration");

        // Window durations are scheduler metadata; effective policy/table alone
        // identify a graph. A control-only launch selects the same frozen pair.
        const auto * control_frozen = &config(control_only);
        moe_fixture control_switched(cpu.get());
        equal_values(control_switched.run(control_frozen->primary, 1, 1, control_frozen->table), pv);
        require(llama_k3_runtime_guard_set(&control_only, true) && llama_k3_runtime_guard_set(&control_only, true),
            "control-only guard activation rejected with zero start/opener budgets");
        equal_values(control_switched.run(control_frozen->guard), gv);
        require(llama_k3_runtime_guard_set(&control_only, false) && llama_k3_runtime_guard_set(&control_only, false),
            "control-only primary restoration failed");
        equal_values(control_switched.run(control_frozen->primary, 1, 1, control_frozen->table), pv);
        require(control_switched.builds == 3 && &config(control_only) == control_frozen &&
            control_frozen->guard_tokens == 0 && control_frozen->guard_start_tokens == 0 &&
            control_frozen->guard_control_tokens == 1 &&
            llama_k3_expert_tables_equal(control_frozen->table, snapshot.table),
            "control-only switch failed to rebuild or changed frozen configuration");

        auto changed = frozen->table; changed[0].mass = .95f;
        moe_fixture cold_mass(cpu.get());
        auto mv = switched.run(frozen->primary, 1, 1, changed);
        equal_values(mv, cold_mass.run(frozen->primary, 1, 1, changed));
        expect_routes(mv, 3, {.5f/.9f, .25f/.9f, .15f/.9f, 0});
        require(switched.builds == 4, "table mass change reused old graph");
        moe_fixture second_layer(cpu.get()), sparse(cpu.get()), sparse_plain(cpu.get());
        equal_values(second_layer.run(frozen->primary, 1, 2, frozen->table), mv);
        equal_values(sparse.run(frozen->primary, 1, 2, {frozen->table[0]}), sparse_plain.run(frozen->primary, 1, 2));
        auto off = frozen->primary; off.mode = llama_k3_decode_expert_mode::disabled;
        moe_fixture table_off(cpu.get()), prefill(cpu.get()), prefill_plain(cpu.get()), outside(cpu.get()), outside_plain(cpu.get());
        equal_values(table_off.run(off, 1, 1, frozen->table), pv);
        equal_values(prefill.run(frozen->primary, 2, 1, frozen->table), prefill_plain.run(off, 2));
        equal_values(outside.run(frozen->primary, 1, 0, frozen->table), outside_plain.run(off, 1, 0));
        auto small = frozen->primary; small.small_batch = true;
        moe_fixture small_batch(cpu.get());
        auto sv = small_batch.run(small, 2, 1, frozen->table);
        require(sv.ids == std::vector<int32_t>({0, 1, -1, -1, 0, 1, -1, -1}), "small-batch opt-in lost");
        auto bounds = frozen->table; bounds[0].mass = 1;
        moe_fixture mass_one(cpu.get()), minimum(cpu.get()), maximum(cpu.get());
        equal_values(mass_one.run(frozen->primary, 1, 1, bounds), mv);
        bounds[0].mass = .1f; bounds[0].min_k = 3;
        equal_values(minimum.run(frozen->primary, 1, 1, bounds), mv);
        bounds[0].mass = .99f; bounds[0].min_k = 1; bounds[0].max_k = 2;
        equal_values(maximum.run(frozen->primary, 1, 1, bounds), pv);

        std::printf("KIMI_RUNTIME_ROUTING_OK launch_snapshot=PASS file_snapshot=PASS guard_bool=PASS "
            "cpu_moe_switch_cold=PASS start_only_switch_cold=PASS start_snapshot_bounds=PASS "
            "control_only_switch_cold=PASS control_snapshot_bounds=PASS "
            "table_parse_rejects=21 comment_semicolons=PASS guard_parse_rejects=25 "
            "graph_key_policy_fields=8 graph_key_table_fields=4 prefill_tail=PASS tokenizer_only=PASS "
            "full_model_kv_kda=not_run gpu=not_run\n");
        return 0;
    } catch (const std::exception & e) {
#if defined(__linux__)
        return test_resources::report("KIMI_RUNTIME_ROUTING", e);
#else
        std::fprintf(stderr, "KIMI_RUNTIME_ROUTING_FAILED: %s\n", e.what());
        return 1;
#endif
    }
}
