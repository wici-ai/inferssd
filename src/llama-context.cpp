#include "llama-context.h"

#include "ggml.h"
#include "llama-arch.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-cpu-sink.h"
#include "llama-deepseek4-hot.h"
#include "llama-host-gather.h"
#include "llama-io.h"
#include "llama-memory.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-ext.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

extern "C" bool llama_moe_prefill_phase_workspace(
        llama_context * ctx,
        int active);
extern "C" bool llama_moe_prefill_phase_workspace_sized(
        llama_context * ctx,
        int active,
        uint32_t requested_tokens);
extern "C" bool llama_moe_prefill_phase_workspace_sized_ex(
        llama_context * ctx,
        int active,
        uint32_t prompt_tokens,
        uint32_t prefill_tokens);

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define LLAMA_MOE_NVTX_HEADER_AVAILABLE 1
#elif __has_include("/usr/local/cuda/include/nvtx3/nvToolsExt.h")
#include "/usr/local/cuda/include/nvtx3/nvToolsExt.h"
#define LLAMA_MOE_NVTX_HEADER_AVAILABLE 1
#else
#define LLAMA_MOE_NVTX_HEADER_AVAILABLE 0
#endif

//
// llama_context
//

namespace {

static std::atomic<uint64_t> kimi_cpu_sink_context_epoch { 1 };

static bool llama_kimi_kda_gpu_layer_selected(int layer) {
    const char * env = std::getenv("GGML_KIMI_KDA_GPU_LAYERS");
    if (env == nullptr || env[0] == '\0' || std::strcmp(env, "0") == 0) {
        return false;
    }
    if (std::strcmp(env, "all") == 0 || std::strcmp(env, "*") == 0) {
        return true;
    }

    const char * p = env;
    while (*p != '\0') {
        while (*p == ' ' || *p == '\t' || *p == ',') {
            ++p;
        }
        if (*p == '\0') {
            break;
        }

        errno = 0;
        char * end = nullptr;
        const long first = std::strtol(p, &end, 10);
        if (errno != 0 || end == p || first < 0 ||
                first > std::numeric_limits<int>::max()) {
            return false;
        }
        long last = first;
        p = end;
        if (*p == '-') {
            ++p;
            errno = 0;
            last = std::strtol(p, &end, 10);
            if (errno != 0 || end == p || last < first ||
                    last > std::numeric_limits<int>::max()) {
                return false;
            }
            p = end;
        }
        if (layer >= first && layer <= last) {
            return true;
        }
        while (*p == ' ' || *p == '\t') {
            ++p;
        }
        if (*p != '\0' && *p != ',') {
            return false;
        }
    }
    return false;
}

struct kimi_logits_trace_state {
    std::mutex mutex;
    FILE * file = nullptr;
    bool initialized = false;
    bool enabled = false;
    bool have_last_hash = false;
    uint64_t sequence = 0;
    uint64_t last_hash = 0;
    int32_t n_vocab = 0;
};

static kimi_logits_trace_state & kimi_logits_trace_get_state() {
    static kimi_logits_trace_state * state = new kimi_logits_trace_state();
    return *state;
}

static uint64_t kimi_logits_trace_hash(const float * logits, int32_t n_vocab) {
    const uint8_t * bytes = reinterpret_cast<const uint8_t *>(logits);
    const size_t size = (size_t) n_vocab * sizeof(float);
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static void kimi_logits_trace_write(const float * logits, int32_t n_vocab, int32_t output_index) {
    const char * path = std::getenv("LLAMA_KIMI_LOGITS_TRACE");
    if (!path || !path[0] || !logits || n_vocab <= 0) {
        return;
    }

    kimi_logits_trace_state & state = kimi_logits_trace_get_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.initialized) {
        state.initialized = true;
        state.file = std::fopen(path, "wb");
        if (!state.file) {
            LLAMA_LOG_ERROR("[kimi_logits_trace] failed to open %s\n", path);
            return;
        }

        const char magic[8] = { 'K', 'L', 'O', 'G', 'I', 'T', 'S', '1' };
        const uint32_t version = 1;
        const uint32_t vocab = (uint32_t) n_vocab;
        if (std::fwrite(magic, sizeof(magic), 1, state.file) != 1 ||
                std::fwrite(&version, sizeof(version), 1, state.file) != 1 ||
                std::fwrite(&vocab, sizeof(vocab), 1, state.file) != 1) {
            LLAMA_LOG_ERROR("[kimi_logits_trace] failed to write header to %s\n", path);
            std::fclose(state.file);
            state.file = nullptr;
            return;
        }
        state.n_vocab = n_vocab;
        state.enabled = true;
        LLAMA_LOG_INFO("[kimi_logits_trace] enabled: %s n_vocab=%d\n", path, n_vocab);
    }

    if (!state.enabled || !state.file || state.n_vocab != n_vocab) {
        return;
    }

    const uint64_t hash = kimi_logits_trace_hash(logits, n_vocab);
    if (state.have_last_hash && state.last_hash == hash) {
        return;
    }

    const uint64_t sequence = state.sequence++;
    const uint32_t vocab = (uint32_t) n_vocab;
    if (std::fwrite(&sequence, sizeof(sequence), 1, state.file) != 1 ||
            std::fwrite(&output_index, sizeof(output_index), 1, state.file) != 1 ||
            std::fwrite(&vocab, sizeof(vocab), 1, state.file) != 1 ||
            std::fwrite(&hash, sizeof(hash), 1, state.file) != 1 ||
            std::fwrite(logits, sizeof(float), (size_t) n_vocab, state.file) != (size_t) n_vocab ||
            std::fflush(state.file) != 0) {
        LLAMA_LOG_ERROR("[kimi_logits_trace] write failed at sequence %llu\n", (unsigned long long) sequence);
        state.enabled = false;
        return;
    }

    state.last_hash = hash;
    state.have_last_hash = true;
}

struct llama_nvtx_api {
    bool initialized = false;
    bool enabled = false;
#if !LLAMA_MOE_NVTX_HEADER_AVAILABLE
    int (*range_push_a)(const char *) = nullptr;
    int (*range_pop)() = nullptr;
#endif
};

static llama_nvtx_api g_llama_nvtx;
static std::once_flag g_llama_nvtx_once;

static void llama_nvtx_init_once() {
    const char *env = std::getenv("GGML_MOE_NVTX");
    if (!env || !env[0] || env[0] == '0') {
        g_llama_nvtx.initialized = true;
        return;
    }
#if LLAMA_MOE_NVTX_HEADER_AVAILABLE
    g_llama_nvtx.enabled = true;
#else
#if !defined(_WIN32)
    void *handle = dlopen("libnvToolsExt.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!handle) {
        handle = dlopen("libnvToolsExt.so", RTLD_LAZY | RTLD_LOCAL);
    }
    if (handle) {
        g_llama_nvtx.range_push_a = reinterpret_cast<int (*)(const char *)>(dlsym(handle, "nvtxRangePushA"));
        g_llama_nvtx.range_pop = reinterpret_cast<int (*)()>(dlsym(handle, "nvtxRangePop"));
        g_llama_nvtx.enabled = g_llama_nvtx.range_push_a && g_llama_nvtx.range_pop;
    }
#endif
#endif
    g_llama_nvtx.initialized = true;
}

static bool llama_nvtx_push(const char *label) {
    std::call_once(g_llama_nvtx_once, llama_nvtx_init_once);
    if (!g_llama_nvtx.enabled || !label || !label[0]) return false;
#if LLAMA_MOE_NVTX_HEADER_AVAILABLE
    nvtxRangePushA(label);
#else
    g_llama_nvtx.range_push_a(label);
#endif
    return true;
}

static void llama_nvtx_pop() {
    if (g_llama_nvtx.enabled) {
#if LLAMA_MOE_NVTX_HEADER_AVAILABLE
        nvtxRangePop();
#else
        if (g_llama_nvtx.range_pop) {
            g_llama_nvtx.range_pop();
        }
#endif
    }
}

class llama_nvtx_range {
public:
    explicit llama_nvtx_range(const char *label) : active(llama_nvtx_push(label)) {}
    ~llama_nvtx_range() {
        if (active) llama_nvtx_pop();
    }
    llama_nvtx_range(const llama_nvtx_range &) = delete;
    llama_nvtx_range & operator=(const llama_nvtx_range &) = delete;

private:
    bool active = false;
};

struct kimi_layer_gap_context_entry {
    uint64_t call_index = 0;
    uint32_t n_tokens = 0;
    int reused = 0;
    uint64_t call_start_us = 0;
    uint64_t graph_prepare_start_us = 0;
    uint64_t graph_build_start_us = 0;
    uint64_t graph_build_end_us = 0;
    uint64_t graph_alloc_end_us = 0;
    uint64_t set_inputs_start_us = 0;
    uint64_t set_inputs_end_us = 0;
    uint64_t compute_start_us = 0;
    uint64_t compute_end_us = 0;
    uint64_t call_end_us = 0;
};

struct kimi_layer_gap_sync_entry {
    uint32_t queued_tokens = 0;
    uint64_t sync_start_us = 0;
    uint64_t sync_end_us = 0;
};

struct kimi_layer_gap_context_buffer {
    std::vector<kimi_layer_gap_context_entry> entries;
    std::vector<kimi_layer_gap_sync_entry> sync_entries;
};

static const char * const g_kimi_layer_gap_context_prefix =
    std::getenv("GGML_MOE_TRACE_LAYER_GAPS");
static const bool g_kimi_layer_gap_context_enabled =
    g_kimi_layer_gap_context_prefix != nullptr && g_kimi_layer_gap_context_prefix[0] != '\0';
static std::atomic<uint64_t> g_kimi_layer_gap_context_call_index{0};
static std::once_flag g_kimi_layer_gap_context_init_once;

static std::mutex & kimi_layer_gap_context_registry_mutex() {
    static auto * mutex = new std::mutex();
    return *mutex;
}

static std::vector<kimi_layer_gap_context_buffer *> & kimi_layer_gap_context_registry() {
    static auto * registry = new std::vector<kimi_layer_gap_context_buffer *>();
    return *registry;
}

static void kimi_layer_gap_context_report_atexit() {
    if (!g_kimi_layer_gap_context_enabled) {
        return;
    }
    const std::string path = std::string(g_kimi_layer_gap_context_prefix) + ".context.csv";
    FILE * file = std::fopen(path.c_str(), "w");
    if (file == nullptr) {
        std::fprintf(stderr, "[kimi_layer_gap] context trace open failed: %s\n", path.c_str());
        return;
    }
    std::fprintf(
        file,
        "call_index,n_tokens,reused,call_start_us,graph_prepare_start_us,"
        "graph_build_start_us,graph_build_end_us,graph_alloc_end_us,"
        "set_inputs_start_us,set_inputs_end_us,compute_start_us,compute_end_us,call_end_us\n");
    size_t rows = 0;
    {
        std::lock_guard<std::mutex> lock(kimi_layer_gap_context_registry_mutex());
        for (const kimi_layer_gap_context_buffer * buffer : kimi_layer_gap_context_registry()) {
            for (const kimi_layer_gap_context_entry & entry : buffer->entries) {
                std::fprintf(
                    file,
                    "%" PRIu64 ",%u,%d,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ","
                    "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ","
                    "%" PRIu64 ",%" PRIu64 "\n",
                    entry.call_index,
                    entry.n_tokens,
                    entry.reused,
                    entry.call_start_us,
                    entry.graph_prepare_start_us,
                    entry.graph_build_start_us,
                    entry.graph_build_end_us,
                    entry.graph_alloc_end_us,
                    entry.set_inputs_start_us,
                    entry.set_inputs_end_us,
                    entry.compute_start_us,
                    entry.compute_end_us,
                    entry.call_end_us);
                ++rows;
            }
        }
    }
    std::fclose(file);
    std::fprintf(stderr, "[kimi_layer_gap] context trace written: %s (%zu rows)\n", path.c_str(), rows);

    const std::string sync_path = std::string(g_kimi_layer_gap_context_prefix) + ".sync.csv";
    file = std::fopen(sync_path.c_str(), "w");
    if (file == nullptr) {
        std::fprintf(stderr, "[kimi_layer_gap] sync trace open failed: %s\n", sync_path.c_str());
        return;
    }
    std::fprintf(file, "queued_tokens,sync_start_us,sync_end_us\n");
    rows = 0;
    {
        std::lock_guard<std::mutex> lock(kimi_layer_gap_context_registry_mutex());
        for (const kimi_layer_gap_context_buffer * buffer : kimi_layer_gap_context_registry()) {
            for (const kimi_layer_gap_sync_entry & entry : buffer->sync_entries) {
                std::fprintf(
                    file,
                    "%u,%" PRIu64 ",%" PRIu64 "\n",
                    entry.queued_tokens,
                    entry.sync_start_us,
                    entry.sync_end_us);
                ++rows;
            }
        }
    }
    std::fclose(file);
    std::fprintf(stderr, "[kimi_layer_gap] sync trace written: %s (%zu rows)\n", sync_path.c_str(), rows);
}

static kimi_layer_gap_context_buffer & kimi_layer_gap_context_thread_buffer() {
    static thread_local kimi_layer_gap_context_buffer * buffer = nullptr;
    if (buffer == nullptr) {
        std::call_once(g_kimi_layer_gap_context_init_once, []() {
            std::atexit(kimi_layer_gap_context_report_atexit);
            std::fprintf(
                stderr,
                "[kimi_layer_gap] context trace enabled: prefix=%s\n",
                g_kimi_layer_gap_context_prefix);
        });
        buffer = new kimi_layer_gap_context_buffer();
        buffer->entries.reserve(256);
        std::lock_guard<std::mutex> lock(kimi_layer_gap_context_registry_mutex());
        kimi_layer_gap_context_registry().push_back(buffer);
    }
    return *buffer;
}

struct kimi_graph_profile_state {
    std::atomic<uint64_t> submit_calls{0};
    std::atomic<uint64_t> submit_us{0};
    std::atomic<uint64_t> sync_calls{0};
    std::atomic<uint64_t> sync_us{0};
    std::atomic<uint64_t> sync_decode_calls{0};
    std::atomic<uint64_t> sync_decode_us{0};
    std::atomic<uint64_t> sync_prompt_calls{0};
    std::atomic<uint64_t> sync_prompt_us{0};
    std::atomic<uint64_t> sync_idle_calls{0};
    std::atomic<uint64_t> sync_idle_us{0};
    std::atomic<uint64_t> sync_decode_tokens{0};
    std::atomic<uint64_t> sync_prompt_tokens{0};
};

static kimi_graph_profile_state g_kimi_graph_profile;

static const char * ds4_weight_profile_path() {
    const char * path = std::getenv("GGML_DS4_WEIGHT_PROFILE_OUT");
    if (path == nullptr || path[0] == '\0' || std::strcmp(path, "0") == 0) {
        return nullptr;
    }
    return path;
}

static FILE * ds4_weight_profile_fp() {
    static FILE * fp = nullptr;
    static bool attempted = false;
    if (fp != nullptr || attempted) {
        return fp;
    }
    attempted = true;
    const char * path = ds4_weight_profile_path();
    if (path == nullptr) {
        return nullptr;
    }
    fp = std::fopen(path, "w");
    if (fp == nullptr) {
        LLAMA_LOG_WARN("ds4 weight profile: failed to open %s\n", path);
        return nullptr;
    }
    std::fprintf(fp, "graph_seq,batched,node,il,ne0,ne1,ne2,nb0,nb1,nb2,token,rank,value\n");
    std::fflush(fp);
    return fp;
}

static int ds4_weight_profile_layer_from_name(const char * name) {
    if (name == nullptr) {
        return -1;
    }
    const char * dash = std::strrchr(name, '-');
    if (dash == nullptr || dash[1] == '\0') {
        return -1;
    }
    char * end = nullptr;
    const long value = std::strtol(dash + 1, &end, 10);
    if (end == dash + 1) {
        return -1;
    }
    return (int) value;
}

static void kimi_graph_profile_report() {
    const uint64_t submit_calls = g_kimi_graph_profile.submit_calls.load();
    const uint64_t sync_calls   = g_kimi_graph_profile.sync_calls.load();
    if (submit_calls == 0 && sync_calls == 0) {
        return;
    }
    const auto avg_ms = [](uint64_t us, uint64_t calls) {
        return calls == 0 ? 0.0 : (double) us / 1000.0 / (double) calls;
    };
    const uint64_t submit_us       = g_kimi_graph_profile.submit_us.load();
    const uint64_t sync_us         = g_kimi_graph_profile.sync_us.load();
    const uint64_t sync_dec_calls  = g_kimi_graph_profile.sync_decode_calls.load();
    const uint64_t sync_dec_us     = g_kimi_graph_profile.sync_decode_us.load();
    const uint64_t sync_pr_calls   = g_kimi_graph_profile.sync_prompt_calls.load();
    const uint64_t sync_pr_us      = g_kimi_graph_profile.sync_prompt_us.load();
    const uint64_t sync_idle_calls = g_kimi_graph_profile.sync_idle_calls.load();
    const uint64_t sync_idle_us    = g_kimi_graph_profile.sync_idle_us.load();
    LLAMA_LOG_INFO(
        "[kimi_graph_profile] submit: calls=%" PRIu64 " total=%.3f ms avg=%.3f ms/call\n",
        submit_calls, (double) submit_us / 1000.0, avg_ms(submit_us, submit_calls));
    LLAMA_LOG_INFO(
        "[kimi_graph_profile] sync: calls=%" PRIu64 " total=%.3f ms avg=%.3f ms/call "
        "decode_calls=%" PRIu64 " decode_tokens=%" PRIu64 " decode_total=%.3f ms decode_avg=%.3f ms/call "
        "prompt_calls=%" PRIu64 " prompt_tokens=%" PRIu64 " prompt_total=%.3f ms prompt_avg=%.3f ms/call "
        "idle_calls=%" PRIu64 " idle_total=%.3f ms idle_avg=%.3f ms/call\n",
        sync_calls, (double) sync_us / 1000.0, avg_ms(sync_us, sync_calls),
        sync_dec_calls, g_kimi_graph_profile.sync_decode_tokens.load(),
        (double) sync_dec_us / 1000.0, avg_ms(sync_dec_us, sync_dec_calls),
        sync_pr_calls, g_kimi_graph_profile.sync_prompt_tokens.load(),
        (double) sync_pr_us / 1000.0, avg_ms(sync_pr_us, sync_pr_calls),
        sync_idle_calls, (double) sync_idle_us / 1000.0, avg_ms(sync_idle_us, sync_idle_calls));
}

static bool kimi_graph_profile_enabled() {
    static bool enabled = []() {
        const char * env = std::getenv("LLAMA_KIMI_GRAPH_PROFILE");
        const bool on = env && env[0] && env[0] != '0';
        if (on) {
            std::atexit(kimi_graph_profile_report);
        }
        return on;
    }();
    return enabled;
}

struct moe_next_gate_shadow_csv_state {
    std::mutex mutex;
    std::ofstream file;
    bool initialized = false;
    bool disabled = false;
    uint64_t call = 0;
};

static moe_next_gate_shadow_csv_state g_moe_next_gate_shadow_csv;

static const char * moe_next_gate_shadow_out_path() {
    const char * path = std::getenv("GGML_MOE_NEXT_GATE_SHADOW_OUT");
    return path && path[0] ? path : nullptr;
}

struct moe_route_score_trace_csv_state {
    std::mutex mutex;
    std::ofstream file;
    bool initialized = false;
    bool disabled = false;
    uint64_t call = 0;
};

static moe_route_score_trace_csv_state g_moe_route_score_trace_csv;

static const char * moe_route_score_trace_out_path() {
    const char * path = std::getenv("GGML_MOE_ROUTE_SCORE_TRACE_OUT");
    return path && path[0] ? path : nullptr;
}

struct moe_next_gate_prefetch_probe_state {
    std::mutex mutex;
    std::ofstream file;
    bool initialized = false;
    bool disabled = false;
    uint64_t seq = 0;
};

static moe_next_gate_prefetch_probe_state g_moe_next_gate_prefetch_probe;

static const char * moe_next_gate_prefetch_probe_out_path() {
    const char * path = std::getenv("GGML_MOE_NEXT_GATE_PREFETCH_PROBE_OUT");
    return path && path[0] ? path : nullptr;
}

static bool moe_next_gate_prefetch_probe_wants(const ggml_tensor * t) {
    const char * path = moe_next_gate_prefetch_probe_out_path();
    if (!path || !t || !t->name[0] || t->type != GGML_TYPE_I32 || !ggml_is_contiguous(t)) {
        return false;
    }
    return std::strstr(t->name, "ffn_moe_next_shadow_topk") != nullptr ||
        std::strstr(t->name, "ffn_moe_topk") != nullptr;
}

static std::string moe_next_gate_shadow_values_csv(const std::vector<int32_t> & values) {
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << '|';
        }
        oss << values[i];
    }
    return oss.str();
}

static std::string moe_route_score_float_values_csv(const std::vector<float> & values) {
    std::ostringstream oss;
    oss.precision(9);
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << '|';
        }
        oss << values[i];
    }
    return oss.str();
}

static void moe_next_gate_prefetch_probe_record(ggml_tensor * t) {
    const char * path = moe_next_gate_prefetch_probe_out_path();
    if (!path || !t || !t->name[0] || t->type != GGML_TYPE_I32 || !ggml_is_contiguous(t)) {
        return;
    }

    const int64_t start_us = ggml_time_us();
    std::vector<int32_t> values((size_t)ggml_nelements(t));
    if (ggml_backend_buffer_is_host(t->buffer)) {
        std::memcpy(values.data(), t->data, ggml_nbytes(t));
    } else {
        ggml_backend_tensor_get(t, values.data(), 0, ggml_nbytes(t));
    }
    const int64_t copy_us = ggml_time_us() - start_us;

    std::lock_guard<std::mutex> lock(g_moe_next_gate_prefetch_probe.mutex);
    if (g_moe_next_gate_prefetch_probe.disabled) {
        return;
    }
    if (!g_moe_next_gate_prefetch_probe.initialized) {
        g_moe_next_gate_prefetch_probe.file.open(path, std::ios::out);
        if (!g_moe_next_gate_prefetch_probe.file) {
            LLAMA_LOG_ERROR("%s: failed to open GGML_MOE_NEXT_GATE_PREFETCH_PROBE_OUT=%s\n", __func__, path);
            g_moe_next_gate_prefetch_probe.disabled = true;
            return;
        }
        g_moe_next_gate_prefetch_probe.file
            << "seq,time_us,name,kind,ne0,ne1,nbytes,copy_us,values\n";
        g_moe_next_gate_prefetch_probe.initialized = true;
    }

    const bool predicted = std::strstr(t->name, "ffn_moe_next_shadow_topk") != nullptr;
    g_moe_next_gate_prefetch_probe.file
        << g_moe_next_gate_prefetch_probe.seq++ << ','
        << ggml_time_us() << ','
        << t->name << ','
        << (predicted ? "pred" : "actual") << ','
        << t->ne[0] << ','
        << t->ne[1] << ','
        << ggml_nbytes(t) << ','
        << copy_us << ','
        << moe_next_gate_shadow_values_csv(values)
        << '\n';
}

static bool moe_next_gate_prefetch_probe_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    llama_context * lctx = (llama_context *) user_data;
    const bool probe_need = moe_next_gate_prefetch_probe_wants(t);
    const llama_cparams * cparams = lctx ? &lctx->get_cparams() : nullptr;
    if (ask) {
        const bool user_need = cparams && cparams->cb_eval ?
            cparams->cb_eval(t, true, cparams->cb_eval_user_data) : false;
        return probe_need || user_need;
    }
    if (probe_need) {
        moe_next_gate_prefetch_probe_record(t);
    }
    if (cparams && cparams->cb_eval) {
        return cparams->cb_eval(t, false, cparams->cb_eval_user_data);
    }
    return true;
}

static void moe_next_gate_shadow_record(
        llm_graph_result * res,
        const llama_ubatch & ubatch,
        ggml_backend_sched_t sched) {
    const auto & outputs = res->get_moe_next_gate_shadow_outputs();
    if (outputs.empty()) {
        return;
    }

    const char * path = moe_next_gate_shadow_out_path();
    if (!path) {
        return;
    }

    struct pending_shadow_copy {
        llm_moe_next_gate_shadow_output meta;
        std::vector<int32_t> values;
        int64_t ne0 = 0;
        int64_t ne1 = 0;
    };

    std::vector<pending_shadow_copy> pending;
    pending.reserve(outputs.size());

    const int64_t copy_start_us = ggml_time_us();
    for (const auto & out : outputs) {
        ggml_tensor * t = out.tensor;
        if (t == nullptr || t->type != GGML_TYPE_I32 || !ggml_is_contiguous(t)) {
            continue;
        }

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, t);
        if (backend == nullptr) {
            continue;
        }

        pending_shadow_copy copy;
        copy.meta = out;
        copy.ne0 = t->ne[0];
        copy.ne1 = t->ne[1];
        copy.values.resize((size_t) ggml_nelements(t));

        ggml_backend_tensor_get_async(backend, t, copy.values.data(), 0, ggml_nbytes(t));
        pending.push_back(std::move(copy));
    }

    if (pending.empty()) {
        return;
    }

    ggml_backend_sched_synchronize(sched);
    const int64_t record_us = ggml_time_us() - copy_start_us;

    std::lock_guard<std::mutex> lock(g_moe_next_gate_shadow_csv.mutex);
    if (g_moe_next_gate_shadow_csv.disabled) {
        return;
    }

    if (!g_moe_next_gate_shadow_csv.initialized) {
        g_moe_next_gate_shadow_csv.file.open(path, std::ios::out);
        if (!g_moe_next_gate_shadow_csv.file) {
            LLAMA_LOG_ERROR("%s: failed to open GGML_MOE_NEXT_GATE_SHADOW_OUT=%s\n", __func__, path);
            g_moe_next_gate_shadow_csv.disabled = true;
            return;
        }
        g_moe_next_gate_shadow_csv.file
            << "call,seq_id,pos,n_tokens,n_seq_tokens,source_layer,target_layer,kind,topk,ne0,ne1,record_us,values\n";
        g_moe_next_gate_shadow_csv.initialized = true;
    }

    const uint64_t call = g_moe_next_gate_shadow_csv.call++;
    const int seq_id = ubatch.n_tokens > 0 && ubatch.seq_id != nullptr && ubatch.seq_id[0] != nullptr ? ubatch.seq_id[0][0] : -1;
    const int64_t pos = ubatch.n_tokens > 0 && ubatch.pos != nullptr ? ubatch.pos[0] : -1;

    for (const auto & copy : pending) {
        g_moe_next_gate_shadow_csv.file
            << call << ','
            << seq_id << ','
            << pos << ','
            << ubatch.n_tokens << ','
            << ubatch.n_seq_tokens << ','
            << copy.meta.source_layer << ','
            << copy.meta.target_layer << ','
            << (copy.meta.predicted ? "pred" : "actual") << ','
            << copy.values.size() << ','
            << copy.ne0 << ','
            << copy.ne1 << ','
            << record_us << ','
            << moe_next_gate_shadow_values_csv(copy.values)
            << '\n';
    }
}

static void moe_route_score_trace_record(
        llm_graph_result * res,
        const llama_ubatch & ubatch,
        ggml_backend_sched_t sched) {
    const auto & outputs = res->get_moe_route_score_outputs();
    if (outputs.empty()) {
        return;
    }

    const char * path = moe_route_score_trace_out_path();
    if (!path) {
        return;
    }

    struct pending_route_score_copy {
        llm_moe_route_score_output meta;
        std::vector<int32_t> ids;
        std::vector<float> scores;
        std::vector<float> weights;
        int64_t ids_ne0 = 0;
        int64_t ids_ne1 = 0;
        int64_t scores_ne0 = 0;
        int64_t scores_ne1 = 0;
        int64_t weights_ne0 = 0;
        int64_t weights_ne1 = 0;
    };

    std::vector<pending_route_score_copy> pending;
    pending.reserve(outputs.size());

    const int64_t copy_start_us = ggml_time_us();
    for (const auto & out : outputs) {
        ggml_tensor * ids = out.ids;
        ggml_tensor * scores = out.scores;
        ggml_tensor * weights = out.weights;
        if (ids == nullptr || scores == nullptr || weights == nullptr) {
            continue;
        }
        if (ids->type != GGML_TYPE_I32 || scores->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32) {
            continue;
        }
        if (!ggml_is_contiguous(ids) || !ggml_is_contiguous(scores) || !ggml_is_contiguous(weights)) {
            continue;
        }

        ggml_backend_t ids_backend = ggml_backend_sched_get_tensor_backend(sched, ids);
        ggml_backend_t scores_backend = ggml_backend_sched_get_tensor_backend(sched, scores);
        ggml_backend_t weights_backend = ggml_backend_sched_get_tensor_backend(sched, weights);
        if (ids_backend == nullptr || scores_backend == nullptr || weights_backend == nullptr) {
            continue;
        }

        pending_route_score_copy copy;
        copy.meta = out;
        copy.ids_ne0 = ids->ne[0];
        copy.ids_ne1 = ids->ne[1];
        copy.scores_ne0 = scores->ne[0];
        copy.scores_ne1 = scores->ne[1];
        copy.weights_ne0 = weights->ne[0];
        copy.weights_ne1 = weights->ne[1];
        copy.ids.resize((size_t) ggml_nelements(ids));
        copy.scores.resize((size_t) ggml_nelements(scores));
        copy.weights.resize((size_t) ggml_nelements(weights));

        ggml_backend_tensor_get_async(ids_backend, ids, copy.ids.data(), 0, ggml_nbytes(ids));
        ggml_backend_tensor_get_async(scores_backend, scores, copy.scores.data(), 0, ggml_nbytes(scores));
        ggml_backend_tensor_get_async(weights_backend, weights, copy.weights.data(), 0, ggml_nbytes(weights));
        pending.push_back(std::move(copy));
    }

    if (pending.empty()) {
        return;
    }

    ggml_backend_sched_synchronize(sched);
    const int64_t record_us = ggml_time_us() - copy_start_us;

    std::lock_guard<std::mutex> lock(g_moe_route_score_trace_csv.mutex);
    if (g_moe_route_score_trace_csv.disabled) {
        return;
    }

    if (!g_moe_route_score_trace_csv.initialized) {
        g_moe_route_score_trace_csv.file.open(path, std::ios::out);
        if (!g_moe_route_score_trace_csv.file) {
            LLAMA_LOG_ERROR("%s: failed to open GGML_MOE_ROUTE_SCORE_TRACE_OUT=%s\n", __func__, path);
            g_moe_route_score_trace_csv.disabled = true;
            return;
        }
        g_moe_route_score_trace_csv.file
            << "call,seq_id,pos,n_tokens,n_seq_tokens,layer,topk,"
            << "ids_ne0,ids_ne1,scores_ne0,scores_ne1,weights_ne0,weights_ne1,"
            << "record_us,ids,scores,weights\n";
        g_moe_route_score_trace_csv.initialized = true;
    }

    const uint64_t call = g_moe_route_score_trace_csv.call++;
    const int seq_id = ubatch.n_tokens > 0 && ubatch.seq_id != nullptr && ubatch.seq_id[0] != nullptr ? ubatch.seq_id[0][0] : -1;
    const int64_t pos = ubatch.n_tokens > 0 && ubatch.pos != nullptr ? ubatch.pos[0] : -1;

    for (const auto & copy : pending) {
        g_moe_route_score_trace_csv.file
            << call << ','
            << seq_id << ','
            << pos << ','
            << ubatch.n_tokens << ','
            << ubatch.n_seq_tokens << ','
            << copy.meta.layer << ','
            << copy.ids.size() << ','
            << copy.ids_ne0 << ','
            << copy.ids_ne1 << ','
            << copy.scores_ne0 << ','
            << copy.scores_ne1 << ','
            << copy.weights_ne0 << ','
            << copy.weights_ne1 << ','
            << record_us << ','
            << moe_next_gate_shadow_values_csv(copy.ids) << ','
            << moe_route_score_float_values_csv(copy.scores) << ','
            << moe_route_score_float_values_csv(copy.weights)
            << '\n';
    }
}

}

// Freeze the launch routing form before any scheduler reserve/runtime graph.
// The only later mutation is the bool selecting primary or guard; no request
// can supply a policy or table through this internal interface.
static llama_k3_runtime_config llama_k3_runtime_from_env(const llama_hparams & hp) {
    llama_k3_runtime_config config;
    config.primary = llama_k3_decode_expert_policy_from_env(hp.n_expert_used, hp.n_layer);
    if (!config.primary.valid()) {
        throw std::invalid_argument(std::string("invalid K3 decode expert policy: ") +
            llama_k3_decode_expert_policy_error_name(config.primary.error));
    }
    config.table = llama_k3_expert_table_from_env(hp.n_expert_used, hp.n_layer);
    config.guard = config.primary;
    config.guard.mode = llama_k3_decode_expert_mode::hard;
    config.guard.top_k = hp.n_expert_used;

    const auto value = [](const char * suffix) {
        const std::string name = std::string("LLAMA_K3_DECODE_EXPERT_GUARD_") + suffix;
        const char * text = std::getenv(name.c_str());
        return text && text[0] ? text : nullptr;
    };
    const auto integer = [&](const char * suffix, int64_t & field, int64_t lower, int64_t upper) {
        const char * text = value(suffix);
        if (text && (!llama_k3_parse_i64_strict(text, field) || field < lower || field > upper)) {
            throw std::invalid_argument(std::string("invalid K3 expert guard ") + suffix);
        }
    };
    int64_t tokens = 0;
    integer("TOKENS", tokens, 0, INT32_MAX);
    config.guard_tokens = uint32_t(tokens);
    int64_t start_tokens = 0;
    integer("START_TOKENS", start_tokens, 0, INT32_MAX);
    config.guard_start_tokens = uint32_t(start_tokens);
    int64_t control_tokens = 0;
    integer("CONTROL_TOKENS", control_tokens, 0, INT32_MAX);
    config.guard_control_tokens = uint32_t(control_tokens);
    if (const char * mode = value("MODE")) {
        if (std::strcmp(mode, "off") == 0) { config.guard.mode = llama_k3_decode_expert_mode::disabled; }
        else if (std::strcmp(mode, "hard") == 0) { config.guard.mode = llama_k3_decode_expert_mode::hard; }
        else if (std::strcmp(mode, "mass") == 0) { config.guard.mode = llama_k3_decode_expert_mode::mass; }
        else { throw std::invalid_argument("invalid K3 expert guard MODE"); }
    }
    integer("TOP_K", config.guard.top_k, 1, hp.n_expert_used);
    integer("MIN_K", config.guard.min_k, 1, hp.n_expert_used);
    integer("MAX_K", config.guard.max_k, 0, hp.n_expert_used);
    if (const char * mass = value("MASS")) {
        if (!llama_k3_parse_f32_strict(mass, config.guard.mass) || config.guard.mass <= 0 || config.guard.mass > 1) {
            throw std::invalid_argument("invalid K3 expert guard MASS");
        }
    }
    if (config.guard.mode == llama_k3_decode_expert_mode::mass &&
            (config.guard.mass <= 0 || config.guard.mass > 1 || config.guard.min_k < 1 ||
             config.guard.min_k > hp.n_expert_used ||
             (config.guard.max_k > 0 && config.guard.min_k > config.guard.max_k))) {
        throw std::invalid_argument("invalid K3 expert guard mass/min_k/max_k combination");
    }
    if (!hp.expert_weights_norm &&
            (config.primary.mode == llama_k3_decode_expert_mode::mass ||
             config.guard.mode == llama_k3_decode_expert_mode::mass || !config.table.empty())) {
        throw std::invalid_argument("K3 cumulative-mass routing requires normalized routing weights");
    }
    return config;
}

llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    cvec(std::make_unique<llama_adapter_cvec>()),
    loras(std::make_unique<llama_adapter_loras>()),
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd())) {
    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;

    // Real vocab-only loaders omit model routing metadata. Preserve those
    // graphless callers; incomplete full-model metadata still fails validation.
    if (model.arch == LLM_ARCH_KIMI_K3 &&
            !(hparams.vocab_only && (hparams.n_expert_used == 0 || hparams.n_layer == 0))) {
        k3_runtime = std::make_unique<const llama_k3_runtime_config>(llama_k3_runtime_from_env(hparams));
    }

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    cparams.n_threads        = params.n_threads;
    cparams.n_threads_batch  = params.n_threads_batch;
    cparams.yarn_ext_factor  = params.yarn_ext_factor  >= 0.0f ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast   = params.yarn_beta_fast   >= 0.0f ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow   = params.yarn_beta_slow   >= 0.0f ? params.yarn_beta_slow   : hparams.yarn_beta_slow;
    cparams.embeddings       = params.embeddings;
    cparams.embeddings_nextn = false;
    cparams.embeddings_nextn_masked = false;
    cparams.offload_kqv      = params.offload_kqv;
    cparams.no_perf          = params.no_perf;
    cparams.pooling_type     = params.pooling_type;
    cparams.warmup           = false;

    cparams.embeddings_layer_inp.resize(hparams.n_layer + 1, false);
    embd_layer_inp.resize(hparams.n_layer + 1);
    cparams.ctx_other = nullptr;

    if (model.arch == LLM_ARCH_DFLASH) {
        if (params.ctx_other == nullptr) {
            throw std::runtime_error("DFlash/DSpark requires ctx_other to pair the draft with its target context");
        }
        cparams.ctx_other = params.ctx_other;
    }

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
            }
        }
    }

    auto rope_scaling_type = params.rope_scaling_type;
    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        static auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };

        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // note: here we assume `mscale == 1.0f`
            // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
                  float mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            // special-case DEEPSEEK v2:
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
            if (model.arch == LLM_ARCH_DEEPSEEK2 && mscale_all_dims != 1.0f) {
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);

            LLAMA_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                    __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
        } else {
            cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/llama.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/llama.cpp/discussions/7416
        //      https://github.com/ggml-org/llama.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;

    cparams.fused_gdn_ar = true;
    cparams.fused_gdn_ch = true;
    cparams.auto_fgdn    = true;

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    const char * dense_gpu_env = std::getenv("GGML_KIMI_DENSE_PREFETCH_LAYERS");
    const char * kda_gpu_env = std::getenv("GGML_KIMI_KDA_GPU_LAYERS");
    dense_gpu_phase_gate_enabled =
        (dense_gpu_env && dense_gpu_env[0] && std::strcmp(dense_gpu_env, "0") != 0) ||
        (kda_gpu_env && kda_gpu_env[0] && std::strcmp(kda_gpu_env, "0") != 0);

    const char * cpu_sink_env = std::getenv("GGML_KIMI_CPU_SINK");
    cpu_sink_enabled = cpu_sink_env && cpu_sink_env[0] && std::strcmp(cpu_sink_env, "0") != 0;
    // The prefill route is decided per ubatch (cpu_sink_prefill_route): the CPU sink's grouped prefill up to a number
    // of tokens, the ggml graph on the GPU above it.  GGML_KIMI_CPU_PREFILL_MAX_TOKENS is that number;
    // GGML_KIMI_CPU_PREFILL=1 and 0 spell "every ubatch" and "none".  The threshold is one input: one of the two.
    const char * cpu_prefill_env = std::getenv("GGML_KIMI_CPU_PREFILL");
    if (cpu_prefill_env && cpu_prefill_env[0] &&
            std::strcmp(cpu_prefill_env, "0") != 0 && std::strcmp(cpu_prefill_env, "1") != 0) {
        throw std::runtime_error("GGML_KIMI_CPU_PREFILL must be 0 or 1");
    }
    cpu_prefill_max_tokens = cpu_prefill_env && std::strcmp(cpu_prefill_env, "1") == 0 ?
        std::numeric_limits<uint32_t>::max() : 0;
    if (const char * value = std::getenv("GGML_KIMI_CPU_PREFILL_MAX_TOKENS"); value && value[0]) {
        if (cpu_prefill_env && cpu_prefill_env[0]) {
            throw std::runtime_error("GGML_KIMI_CPU_PREFILL and GGML_KIMI_CPU_PREFILL_MAX_TOKENS are both set: the "
                "prefill threshold is one input");
        }
        char * end = nullptr;
        errno = 0;
        const unsigned long long parsed = std::strtoull(value, &end, 10);
        if (errno != 0 || !std::isdigit((unsigned char) value[0]) || *end != '\0' ||
                parsed > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("GGML_KIMI_CPU_PREFILL_MAX_TOKENS must be a number of tokens");
        }
        cpu_prefill_max_tokens = uint32_t(parsed);
    }
    cpu_prefill_request_max_tokens = cpu_prefill_max_tokens;
    if (cpu_prefill_max_tokens > 0) {
        if (!cpu_sink_enabled) {
            throw std::runtime_error("GGML_KIMI_CPU_PREFILL=1 or GGML_KIMI_CPU_PREFILL_MAX_TOKENS > 0 requires GGML_KIMI_CPU_SINK=1");
        }
        if (model.arch != LLM_ARCH_KIMI_K3 && model.arch != LLM_ARCH_DEEPSEEK41 && model.arch != LLM_ARCH_MUSE_SPARK) {
            throw std::runtime_error("GGML_KIMI_CPU_PREFILL supports only Kimi-K3, DeepSeek-V4.1 or Muse Spark models; set GGML_KIMI_CPU_PREFILL=0");
        }
    }
    if (cpu_prefill_max_tokens == std::numeric_limits<uint32_t>::max()) {
        // The grouped CPU executor has its own small bounded workspace. With every
        // prefill ubatch on it, the legacy GPU union workspace would serve none:
        // do not allocate it alongside.  Below a finite threshold the workspace
        // serves the graph route (K3's phase-workspace profiles with the sink on).
        for (const char * name : { "GGML_MOE_PREFILL_PHASE0", "GGML_MOE_PREFILL_PHASE1",
                "GGML_MOE_PREFILL_PHASE_SIZED_WORKSPACE", "GGML_MOE_PREFILL_DECODE_CACHE_HANDOVER" }) {
            const char * value = std::getenv(name);
            if (value && value[0] && std::strcmp(value, "0") != 0) {
                throw std::runtime_error(std::string("CPU prefill requires ") + name + "=0");
            }
        }
    }
    if (cpu_sink_enabled) {
        // with the sink on, any prefill ubatch may take its grouped prefill, whatever the threshold: one the GPU does
        // not offload, or a whole request whose graph does not fit
        if (cparams.n_ubatch > uint32_t(std::numeric_limits<int>::max() / 16)) {
            throw std::runtime_error("grouped CPU prefill ubatch exceeds route index range");
        }
        LLAMA_LOG_INFO("%s: grouped CPU prefill configured; ubatch=%u\n",
            __func__, cparams.n_ubatch);
    }
    if (cpu_sink_enabled) {
        if (model.arch != LLM_ARCH_KIMI_K3 && model.arch != LLM_ARCH_DEEPSEEK41 && model.arch != LLM_ARCH_MUSE_SPARK) {
            throw std::runtime_error("GGML_KIMI_CPU_SINK requires a Kimi-K3, DeepSeek-V4.1 or Muse Spark model");
        }
        if (cparams.n_seq_max != 1) {
            throw std::runtime_error("GGML_KIMI_CPU_SINK requires --parallel 1 (-np 1): request phase is context-wide");
        }
#if !defined(__linux__)
        throw std::runtime_error("GGML_KIMI_CPU_SINK requires Linux io_uring and O_DIRECT");
#endif
        // Layers with routed experts, in model order: the default cache priority
        // outside the K3 layout. The routed expert count bounds cache quotas.
        std::vector<int32_t> routed_layers;
        for (uint32_t il = 0; il < model.hparams.n_layer; ++il) {
            const auto & layer = model.layers[il];
            if (layer.ffn_up_exps && layer.ffn_gate_exps && layer.ffn_down_exps) {
                routed_layers.push_back(int32_t(il));
            }
        }
        cpu_sink_runtime = std::make_unique<llama_kimi_cpu_sink_runtime>(model.hparams.n_layer,
            kimi_cpu_sink_context_epoch.fetch_add(1, std::memory_order_relaxed),
            model.arch == LLM_ARCH_KIMI_K3, int64_t(model.hparams.n_expert), routed_layers);
        LLAMA_LOG_INFO("%s: CPU sink configured; single-token decode%s uses CPU experts\n",
            __func__, cpu_prefill_max_tokens > 0 ? " and grouped prefill" : "");
        if (cpu_prefill_max_tokens == std::numeric_limits<uint32_t>::max()) {
            LLAMA_LOG_INFO("%s: prefill route: every ubatch on the CPU sink\n", __func__);
        } else {
            LLAMA_LOG_INFO("%s: prefill route: ubatches of up to %u tokens, and those the GPU does not offload, on "
                "the CPU sink; larger ones on the ggml graph\n", __func__, cpu_prefill_max_tokens);
        }
        // the sink's next-layer prefetch resolves experts of a layer other than the active one
        if (const char * prefetch = std::getenv("GGML_KIMI_CPU_SINK_PREFETCH"); prefetch && std::atoi(prefetch) > 0) {
            for (uint32_t il = 0; il < model.hparams.n_layer; ++il) {
                const auto & layer = model.layers[il];
                for (const ggml_tensor * experts : { layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps }) {
                    if (experts) {
                        cpu_sink_runtime->expert_shapes[ggml_get_name(experts)] = { experts->nb[2], experts->ne[2] };
                    }
                }
            }
        }
    }

    const char * reserve_horizon_env = std::getenv("LLAMA_RESERVE_HORIZON");
    if (reserve_horizon_env && reserve_horizon_env[0] &&
            std::strcmp(reserve_horizon_env, "0") != 0 && std::strcmp(reserve_horizon_env, "1") != 0) {
        throw std::runtime_error("LLAMA_RESERVE_HORIZON must be 0 or 1");
    }
    reserve_horizon_enabled = !(reserve_horizon_env && std::strcmp(reserve_horizon_env, "0") == 0);

    const char * actual_shape_env = std::getenv("GGML_KIMI_CPU_PREFILL_ACTUAL_SHAPE");
    if (actual_shape_env && actual_shape_env[0] &&
            std::strcmp(actual_shape_env, "0") != 0 && std::strcmp(actual_shape_env, "1") != 0) {
        throw std::runtime_error("GGML_KIMI_CPU_PREFILL_ACTUAL_SHAPE must be 0 or 1");
    }
    // Graphs sized to the request on either route: each request reserves its prefill graph at its own size (the
    // route of each pass follows from that size) and leaves only the one-token graph to decode.
    cpu_prefill_actual_shape_enabled = actual_shape_env && std::strcmp(actual_shape_env, "1") == 0;
    if (cpu_prefill_actual_shape_enabled) {
        if (!cpu_sink_enabled) {
            throw std::runtime_error("GGML_KIMI_CPU_PREFILL_ACTUAL_SHAPE requires GGML_KIMI_CPU_SINK=1 for phase transitions");
        }
        cpu_prefill_configured_ubatch = cparams.n_ubatch;
        if (const char * value = std::getenv("GGML_KIMI_CPU_PREFILL_GRAPH_GRANULARITY")) {
            char * end = nullptr;
            errno = 0;
            const unsigned long parsed = std::strtoul(value, &end, 10);
            if (errno != 0 || end == value || *end != '\0' || parsed == 0 || parsed > 2048) {
                throw std::runtime_error("GGML_KIMI_CPU_PREFILL_GRAPH_GRANULARITY must be in 1..2048");
            }
            cpu_prefill_graph_granularity = uint32_t(parsed);
        }
        // Keep the configured ubatch through context construction: the
        // DeepSeek recurrent-memory backend uses it to establish the maximum
        // physical batch capacity.  The first request replaces the initial
        // scheduler with its actual prefill shape, and the prefill->decode
        // transition then leaves only the one-token scheduler resident.
        LLAMA_LOG_INFO(
            "%s: actual-shape grouped prefill enabled; configured_max=%u granularity=%u "
            "resident_decode_ubatch=1 after_first_prefill\n",
            __func__, cpu_prefill_configured_ubatch, cpu_prefill_graph_granularity);
    }

    const char * reserve_outputs_env = std::getenv("LLAMA_PREFILL_GRAPH_OUTPUTS");
    if (reserve_outputs_env && reserve_outputs_env[0] &&
            std::strcmp(reserve_outputs_env, "0") != 0 && std::strcmp(reserve_outputs_env, "1") != 0) {
        throw std::runtime_error("LLAMA_PREFILL_GRAPH_OUTPUTS must be 0 or 1");
    }
    cpu_prefill_output_sizing_enabled = (model.arch == LLM_ARCH_DEEPSEEK41 || model.arch == LLM_ARCH_MUSE_SPARK) &&
        !(reserve_outputs_env && std::strcmp(reserve_outputs_env, "0") == 0);
    if (cpu_prefill_output_sizing_enabled) {
        sched_reserve_n_outputs = cparams.n_seq_max;
        LLAMA_LOG_INFO("%s: prefill output reserve follows actual outputs\n", __func__);
    }

    const char * phase_workspace_env =
        std::getenv("GGML_MOE_PREFILL_PHASE_SIZED_WORKSPACE");
    phase_workspace_enabled = phase_workspace_env && phase_workspace_env[0] &&
        phase_workspace_env[0] != '0';
    const char * phase_decode_cache_handover_env =
        std::getenv("GGML_MOE_PREFILL_DECODE_CACHE_HANDOVER");
    phase_decode_cache_handover_enabled = phase_decode_cache_handover_env &&
        phase_decode_cache_handover_env[0] &&
        phase_decode_cache_handover_env[0] != '0';
    if (phase_decode_cache_handover_enabled && !phase_workspace_enabled) {
        throw std::runtime_error(
            "prefill/decode cache handover requires phase-sized prefill workspace");
    }
    const char * handover_phase0_env = std::getenv("GGML_MOE_PREFILL_PHASE0");
    const char * handover_phase1_env = std::getenv("GGML_MOE_PREFILL_PHASE1");
    if (phase_decode_cache_handover_enabled &&
            (!handover_phase0_env || !handover_phase0_env[0] ||
             handover_phase0_env[0] == '0' ||
             !handover_phase1_env || !handover_phase1_env[0] ||
             handover_phase1_env[0] == '0')) {
        throw std::runtime_error(
            "prefill/decode cache handover requires Phase 0/1 grouped prefill");
    }
    if (phase_workspace_enabled) {
        phase_workspace_target_ubatch = cparams.n_ubatch;
        if ((phase_workspace_target_ubatch != 32 &&
                phase_workspace_target_ubatch != 64 &&
                phase_workspace_target_ubatch != 96) ||
                cparams.n_batch < phase_workspace_target_ubatch) {
            throw std::runtime_error(
                "phase-sized prefill workspace requires matching -b/-ub at capacity level 32, 64, or 96");
        }
        // Keep only the decode-sized graph resident between requests.  The
        // target prefill graph is created transactionally at request begin.
        cparams.n_ubatch = 1;
    }

    cparams.op_offload = params.op_offload;
    cparams.kv_unified = params.kv_unified;

    // initialized later
    cparams.pipeline_parallel = false;

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }
    }

    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx =  cparams.n_ctx_seq * cparams.n_seq_max;
            LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    LLAMA_LOG_INFO("%s: n_seq_max     = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx         = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_seq     = %u\n",   __func__, cparams.n_ctx_seq);
    LLAMA_LOG_INFO("%s: n_batch       = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch      = %u\n",   __func__, cparams.n_ubatch);
    if (phase_workspace_enabled) {
        LLAMA_LOG_INFO(
            "%s: phase workspace target ubatch = %u, resident decode ubatch = %u\n",
            __func__, phase_workspace_target_ubatch, cparams.n_ubatch);
    }
    LLAMA_LOG_INFO("%s: causal_attn   = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn    = %s\n",   __func__, llama_flash_attn_type_name(params.flash_attn_type));
    LLAMA_LOG_INFO("%s: kv_unified    = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
    LLAMA_LOG_INFO("%s: freq_base     = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale    = %g\n",   __func__, cparams.rope_freq_scale);

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (!hparams.vocab_only) {
        // GPU backends
        for (const auto & dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev.dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev.dev)));
            }
            backends.emplace_back(backend);
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            if (output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    // init the memory module
    if (!hparams.vocab_only) {
        llama_memory_params params_mem = {
            /*.type_k   =*/ params.type_k,
            /*.type_v   =*/ params.type_v,
            /*.swa_full =*/ params.swa_full,
        };

        memory.reset(model.create_memory(params_mem, cparams));
    }

    // init backends
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();

        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                const auto & dev = model.devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev.dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
            backend_buf_exp_size.push_back(0);
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.n_gpu_layers() > model.hparams.n_layer &&
            model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv &&
            !model.has_tensor_overrides();

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    // TODO: should we ignore ACCEL types too?
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
        }

        // DeepSeek4 hot-expert pinning: load profile and allocate per-layer
        // hot subset tensors before the first sched_reserve so the graph
        // builder can see them. No-op if DS4_HOT_PROFILE_JSON is unset.
        if (model.arch == LLM_ARCH_DEEPSEEK4) {
            auto & ds4_hot_mgr = ds4_hot::instance();
            if (ds4_hot_mgr.load_profile()) {
                // Pass n_expert_used (P) so the manager can allocate the
                // K + P + 1 dummy/padding slots needed by the dispatch graph.
                ds4_hot_mgr.set_n_picks((int) model.hparams.n_expert_used);
                ds4_hot_mgr.allocate(model);
            }
        }

        const char * moe_phase0_env = std::getenv("GGML_MOE_PREFILL_PHASE0");
        const char * moe_phase1_env = std::getenv("GGML_MOE_PREFILL_PHASE1");
        const bool moe_phase01_requested =
            (moe_phase0_env && moe_phase0_env[0] && moe_phase0_env[0] != '0') ||
            (moe_phase1_env && moe_phase1_env[0] && moe_phase1_env[0] != '0');
#if !defined(_WIN32)
        if (moe_phase01_requested) {
            using moe_phase01_plan_fn = bool (*)(uint32_t, uint32_t);
            auto * moe_phase01_plan = reinterpret_cast<moe_phase01_plan_fn>(
                dlsym(RTLD_DEFAULT, "ggml_cuda_moe_prefill_phase01_plan"));
            if (!moe_phase01_plan) {
                throw std::runtime_error("Phase 0/1 prefill planner is unavailable");
            }
            const uint32_t planner_ubatch = phase_workspace_enabled ?
                phase_workspace_target_ubatch : cparams.n_ubatch;
            if (!moe_phase01_plan(planner_ubatch, hparams.n_expert_used)) {
                throw std::runtime_error("Phase 0/1 prefill planner rejected the graph reserve");
            }
        }
#else
        if (moe_phase01_requested) {
            throw std::runtime_error("Phase 0/1 prefill planner is unsupported on this platform");
        }
#endif

        // Nothing is cached yet: the first reserve covers one ubatch of positions and
        // grows with the context (decode) or is sized per request (actual-shape prefill).
        reserve_horizon_min = std::max<uint32_t>(1, std::min(cparams.n_ubatch, cparams.n_ctx));
        sched_reserve_n_ctx = reserve_horizon_bucket(1);

        sched_reserve();

        // The expert caches' storage once the worst-case graph is reserved (the GPU tier's on unified memory, the RAM
        // cache's elsewhere when it can give room back; the sink decides): its page population then leaves every
        // request's path. Any routed layer gives the experts' format.
        if (cpu_sink_runtime && (cpu_sink_runtime->vram_budget || cpu_sink_runtime->cache_budget) && backend_cpu) {
            for (const auto & layer : model.layers) {
                if (layer.ffn_up_exps && layer.ffn_gate_exps && layer.ffn_down_exps) {
                    cpu_sink_runtime->reserve_tier(ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu)),
                        layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps, int(model.hparams.n_expert_used));
                    break;
                }
            }
        }

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }
    }

    // Initialize the full vocabulary token ids for backend samplers.
    {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }
}

llama_context::~llama_context() {
    {
        ggml_backend_sched_kimi_copy_stats now = {};
        ggml_backend_sched_kimi_copy_stats_get(&now);
        const std::string line = dense_stream.end(now);
        if (!line.empty()) {
            LLAMA_LOG_INFO("%s\n", line.c_str());
        }
    }
    // Graph nodes borrow this context's runtime. Complete their work before
    // releasing cache pages, worker threads, and the backend API pointers.
    if (cpu_sink_runtime) {
        if (sched) { synchronize(); }
        cpu_sink_runtime.reset();
    }
    if (!model.hparams.no_alloc) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            const size_t size_exp = backend_buf_exp_size[i];
            const size_t size_act = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size_exp == size_act) {
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            } else {
                LLAMA_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            }
        }
    }
    ggml_opt_free(opt_ctx);
}

void llama_context::dense_graph_cache_begin_phase(int phase) {
    if (!dense_gpu_phase_gate_enabled && !phase_decode_cache_handover_enabled) {
        return;
    }

    using graph_cache_begin_phase_fn = void (*)(ggml_backend_t, int);
    for (const auto & backend : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg == nullptr) {
            continue;
        }
        auto begin_phase = reinterpret_cast<graph_cache_begin_phase_fn>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_graph_cache_begin_phase"));
        if (begin_phase != nullptr) {
            begin_phase(backend.get(), phase);
        }
    }
    dense_gpu_scheduler_graph_phase = phase;
}

void llama_context::dense_graph_cache_end_phase() {
    if ((!dense_gpu_phase_gate_enabled && !phase_decode_cache_handover_enabled) ||
            dense_gpu_scheduler_graph_phase == 0) {
        return;
    }

    const int phase = dense_gpu_scheduler_graph_phase;
    const bool retain = phase == 2;
    using graph_cache_end_phase_fn = void (*)(ggml_backend_t, int, bool);
    for (const auto & backend : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg == nullptr) {
            continue;
        }
        auto end_phase = reinterpret_cast<graph_cache_end_phase_fn>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_graph_cache_end_phase"));
        if (end_phase != nullptr) {
            end_phase(backend.get(), phase, retain);
        }
    }
    dense_gpu_scheduler_graph_phase = 0;
}

static uint32_t kimi_cpu_sink_reserve_tokens(const llama_cparams & params, bool decode) {
    const char * value = std::getenv("GGML_KIMI_CPU_SINK_DECODE_RESERVE");
    // CPU sink only executes one token at a time. Reserving the ubatch-sized
    // fallback graph during decode needlessly allocates GPU expert workspace.
    // Keep the configured ubatch intact so the next prefill is unchanged.
    if (decode && params.n_seq_max == 1 && value && std::strcmp(value, "1") == 0) {
        return 1;
    }
    return std::min(params.n_ctx, params.n_ubatch);
}

llama_memory_context_ptr llama_context::memory_init_reserve() {
    if (sched_reserve_n_ctx == 0 || sched_reserve_n_ctx >= cparams.n_ctx) {
        return memory->init_full();
    }
    return memory->init_reserve(sched_reserve_n_ctx);
}

uint32_t llama_context::memory_n_pos() const {
    if (!memory) {
        return 0;
    }
    llama_pos pos_max = -1;
    for (llama_seq_id s = 0; s < (llama_seq_id) cparams.n_seq_max; ++s) {
        pos_max = std::max(pos_max, memory->seq_pos_max(s));
    }
    return (uint32_t) (pos_max + 1);
}

uint32_t llama_context::reserve_horizon_bucket(uint32_t n_pos) const {
    if (!reserve_horizon_enabled) {
        return 0;
    }
    uint64_t horizon = std::max<uint32_t>(1, reserve_horizon_min);
    while (horizon < n_pos && horizon < cparams.n_ctx) {
        horizon *= 2;
    }
    return horizon >= cparams.n_ctx ? 0 : (uint32_t) horizon;
}

void llama_context::sched_reserve() {
    if (!sched_need_reserve) {
        return;
    }

    sched_need_reserve = false;

    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);

    synchronize();

    const int64_t t_start_us = ggml_time_us();

    const uint32_t n_seqs = cparams.n_seq_max;
    const uint32_t n_tokens = kimi_cpu_sink_reserve_tokens(cparams,
        cpu_sink_enabled && cpu_sink_decode_active && !cparams.warmup);
    if (n_tokens < std::min(cparams.n_ctx, cparams.n_ubatch)) {
        LLAMA_LOG_INFO("[kimi_cpu_sink_decode_reserve] tokens=%u configured_ubatch=%u\n",
            n_tokens, cparams.n_ubatch);
    }

    const size_t max_nodes = this->graph_max_nodes(n_tokens);

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

    gf_res_prev.reset(new llm_graph_result(max_nodes));
    gf_res_reserve.reset(new llm_graph_result(max_nodes));

    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, cparams.pipeline_parallel, cparams.op_offload));

    llama_memory_context_ptr mctx;
    if (memory) {
        LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
        mctx = memory_init_reserve();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
        if (reserve_horizon_enabled) {
            LLAMA_LOG_INFO("[reserve_horizon] n_ctx_reserve=%u n_ctx=%u n_tokens=%u\n",
                sched_reserve_n_ctx ? sched_reserve_n_ctx : cparams.n_ctx, cparams.n_ctx, n_tokens);
        }
    }

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs = n_seqs;

    LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs);

    // resolve automatic Flash Attention use
    if (cparams.auto_fa) {
        auto * gf = graph_reserve(1, n_seqs, n_outputs, mctx.get(), true);
        if (!gf) {
            throw std::runtime_error("failed to reserve graph for Flash Attention check");
        }

        const size_t prefix_len = strlen(LLAMA_TENSOR_NAME_FATTN) + 1;
        bool fa_device_mismatch = false;
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor * n = ggml_graph_node(gf, i);
            if (n->op != GGML_OP_FLASH_ATTN_EXT) {
                continue;
            }
            ggml_backend_dev_t device_fa = ggml_backend_get_device(ggml_backend_sched_get_tensor_backend(sched.get(), n));

            // TODO: instead of the tensor names, use a map to keep track of which (FA) tensors belong to which layer
            GGML_ASSERT(strncmp(n->name, LLAMA_TENSOR_NAME_FATTN "-", prefix_len) == 0);
            const int il = std::stoi(n->name + prefix_len);
            ggml_backend_dev_t device_kv = model.dev_layer(il);
            if (device_fa != device_kv) {
                LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but the Flash Attention tensor "
                        "is assigned to device %s (usually due to missing support)\n",
                        __func__, il, ggml_backend_dev_name(device_kv), ggml_backend_dev_name(device_fa));
                // FIXME: fa_device_mismatch logic is wrong for --no-kv-offload, but this is broken anyways
                fa_device_mismatch = true;
                break;
            }
        }

        if (fa_device_mismatch) {
            cparams.flash_attn = false;
            LLAMA_LOG_WARN("%s: Flash Attention was auto, set to disabled\n", __func__);
        } else {
            cparams.flash_attn = true;
            LLAMA_LOG_INFO("%s: Flash Attention was auto, set to enabled\n", __func__);
        }

        cparams.auto_fa = false;
    }

    if (cparams.auto_fgdn) {
        LLAMA_LOG_INFO("%s: resolving fused Gated Delta Net support:\n", __func__);

        if (model.arch == LLM_ARCH_DEEPSEEK4 || model.arch == LLM_ARCH_DEEPSEEK41) {
            cparams.fused_gdn_ar = false;
            cparams.fused_gdn_ch = false;
        }

        if (cparams.fused_gdn_ar) {
            auto * gf = graph_reserve(1, n_seqs, n_outputs, mctx.get(), true);
            if (!gf) {
                throw std::runtime_error("failed to reserve graph for fused Gated Delta Net check (autoregressive)");
            }

            const size_t prefix_len = strlen(LLAMA_TENSOR_NAME_FGDN_AR) + 1;
            bool gdn_device_mismatch = false;
            for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
                ggml_tensor * n = ggml_graph_node(gf, i);
                if (n->op != GGML_OP_GATED_DELTA_NET) {
                    continue;
                }
                ggml_backend_dev_t device_gdn = ggml_backend_get_device(ggml_backend_sched_get_tensor_backend(sched.get(), n));

                GGML_ASSERT(strncmp(n->name, LLAMA_TENSOR_NAME_FGDN_AR "-", prefix_len) == 0);
                const int il = std::stoi(n->name + prefix_len);
                ggml_backend_dev_t device_kv = model.dev_layer(il);
                const bool kda_gpu_override = model.arch == LLM_ARCH_KIMI_K3 &&
                    llama_kimi_kda_gpu_layer_selected(il) &&
                    ggml_backend_dev_type(device_gdn) == GGML_BACKEND_DEVICE_TYPE_GPU;
                if (device_gdn != device_kv && !kda_gpu_override) {
                    LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but the fused Gated Delta Net tensor "
                            "is assigned to device %s (usually due to missing support)\n",
                            __func__, il, ggml_backend_dev_name(device_kv), ggml_backend_dev_name(device_gdn));
                    gdn_device_mismatch = true;
                    break;
                }
            }

            if (gdn_device_mismatch) {
                cparams.fused_gdn_ar = false;
                LLAMA_LOG_WARN("%s: fused Gated Delta Net (autoregressive) not supported, set to disabled\n", __func__);
            } else {
                LLAMA_LOG_INFO("%s: fused Gated Delta Net (autoregressive) enabled\n", __func__);
            }
        }

        if (cparams.fused_gdn_ch) {
            // more than one token in the batch per sequence in order to take the chunked path
            // note: n_outputs must match n_tokens for embedding models with mean/rank pooling,
            // because build_pooling creates inp_mean with shape [n_tokens, n_seqs] and multiplies
            // it with t_embd which is reduced to [n_outputs, ...] via out_ids. if n_outputs != n_tokens,
            // the ggml_mul_mat assertion fails. this matches the pp reservation below (line ~553).
            const uint32_t n_tokens_ch = 16*n_seqs;
            auto * gf = graph_reserve(n_tokens_ch, n_seqs, n_tokens_ch, mctx.get(), true);
            if (!gf) {
                throw std::runtime_error("failed to reserve graph for fused Gated Delta Net check (chunked)");
            }

            const size_t prefix_len = strlen(LLAMA_TENSOR_NAME_FGDN_CH) + 1;
            bool gdn_device_mismatch = false;
            for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
                ggml_tensor * n = ggml_graph_node(gf, i);
                if (n->op != GGML_OP_GATED_DELTA_NET) {
                    continue;
                }
                ggml_backend_dev_t device_gdn = ggml_backend_get_device(ggml_backend_sched_get_tensor_backend(sched.get(), n));

                GGML_ASSERT(strncmp(n->name, LLAMA_TENSOR_NAME_FGDN_CH "-", prefix_len) == 0);
                const int il = std::stoi(n->name + prefix_len);
                ggml_backend_dev_t device_kv = model.dev_layer(il);
                const bool kda_gpu_override = model.arch == LLM_ARCH_KIMI_K3 &&
                    llama_kimi_kda_gpu_layer_selected(il) &&
                    ggml_backend_dev_type(device_gdn) == GGML_BACKEND_DEVICE_TYPE_GPU;
                if (device_gdn != device_kv && !kda_gpu_override) {
                    LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but the fused Gated Delta Net tensor "
                            "is assigned to device %s (usually due to missing support)\n",
                            __func__, il, ggml_backend_dev_name(device_kv), ggml_backend_dev_name(device_gdn));
                    gdn_device_mismatch = true;
                    break;
                }
            }

            if (gdn_device_mismatch) {
                cparams.fused_gdn_ch = false;
                LLAMA_LOG_WARN("%s: fused Gated Delta Net (chunked) not supported, set to disabled\n", __func__);
            } else {
                LLAMA_LOG_INFO("%s: fused Gated Delta Net (chunked) enabled\n", __func__);
            }
        }

        cparams.auto_fgdn = false;
    }

    // A sampled prompt normally needs one output per sequence. Requests that
    // emit more outputs grow this reserve before building their next ubatch.
    // Pooled embeddings need all token outputs for their reduction graph.
    const uint32_t n_outputs_pp = cpu_prefill_output_sizing_enabled && !cparams.embeddings ?
        std::min(n_tokens, std::max(n_seqs, sched_reserve_n_outputs)) : n_tokens;
    if (cpu_prefill_output_sizing_enabled) {
        LLAMA_LOG_INFO("[prefill_reserve_outputs] n_tokens=%u n_outputs=%u\n", n_tokens, n_outputs_pp);
    }

    // reserve worst-case graph
    int n_splits_pp = -1;
    int n_nodes_pp  = -1;

    int n_splits_tg = -1;
    int n_nodes_tg  = -1;

    // Host compute buffers (pinned graph inputs, e.g. the attention masks of a long context) come out of
    // the same host memory as the expert caches; measure them first and let the host-memory ledger make room
    // instead of the cgroup or the kernel killing the process. The device buffers are allocated first: on a
    // unified-memory GPU they come out of the same DRAM, so the ledger counts them there too, and neither
    // allocation can take the room made for the other. A prefill's graph also gathers rows of weights the file
    // mapping backs (token embeddings, Engram tables): their pages come into the page cache while it runs, and the
    // ledger holds room for them from here to the first decode token, so that the expert caches give back what the
    // prefill needs at once, before it allocates anything.
    if (cpu_sink_runtime && (cpu_sink_runtime->cache_budget || cpu_sink_runtime->vram_budget) && !model.hparams.no_alloc) {
        std::vector<size_t> sizes(backend_ptrs.size(), 0);
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), true, sizes.data());
        size_t host_bytes = 0, device_bytes = 0;
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            (ggml_backend_buft_is_host(backend_buft[i]) ? host_bytes : device_bytes) += sizes[i];
        }
        const bool prefill = n_tokens > n_seqs && !(cpu_sink_enabled && cpu_sink_decode_active);
        if (prefill) {
            const uint64_t pages = gf ? llama_host_gather_bytes(gf, llama_host_page_size(),
                [&](const ggml_tensor * t) { return model.file_mapped(t); }) : 0;
            LLAMA_LOG_INFO("[prefill_room] n_tokens=%u gather_bytes=%" PRIu64 " host_bytes=%zu device_bytes=%zu\n",
                n_tokens, pages, host_bytes, device_bytes);
            if (!cpu_sink_runtime->prefill_room(pages, host_bytes, device_bytes)) {
                LLAMA_LOG_WARN("%s: the host-memory ledger found no room for a prefill: %.2f MiB of gathered pages, %.2f MiB of host and %.2f MiB of device compute buffers\n",
                    __func__, pages/1024.0/1024.0, host_bytes/1024.0/1024.0, device_bytes/1024.0/1024.0);
            }
        } else if (host_bytes + device_bytes > 0 && !cpu_sink_runtime->make_room(host_bytes, device_bytes)) {
            LLAMA_LOG_WARN("%s: the host-memory ledger found no room for %.2f MiB of host and %.2f MiB of device compute buffers\n",
                __func__, host_bytes/1024.0/1024.0, device_bytes/1024.0/1024.0);
        }
    }

    // reserve pp (prompt processing) graph first so that buffers are only allocated once
    {
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(),
                model.hparams.no_alloc, model.hparams.no_alloc ? backend_buf_exp_size.data() : nullptr);
        if (!gf) {
            if (cparams.pipeline_parallel) {
                LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
                cparams.pipeline_parallel = false;
                sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false, cparams.op_offload));
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get());
            }
            if (!gf) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        n_splits_pp = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_pp  = ggml_graph_n_nodes(gf);
    }

    // reserve with tg (token generation) graph to get the number of splits and nodes
    {
        auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute tg buffers");
        }

        n_splits_tg = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_tg  = ggml_graph_n_nodes(gf);
    }

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    {
        // TODO: not sure if the following graph would be worst case for multi-stream KV caches:
        //
        // auto * gf = graph_reserve(n_tokens, 1, n_tokens, mctx.get());
        //
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute pp buffers");
        }
    }

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!model.hparams.no_alloc) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
        if (backend_buf_exp_size[i] > 1) {
            LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buft_name(buft),
                    backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
    }
    if (cpu_prefill_output_sizing_enabled) {
        sched_reserve_n_outputs = n_outputs_pp;
    }

    if (n_nodes_pp == n_nodes_tg) {
        LLAMA_LOG_INFO("%s: graph nodes  = %d\n", __func__, n_nodes_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph nodes  = %d (with bs=%d), %d (with bs=1)\n", __func__, n_nodes_pp, n_tokens, n_nodes_tg);
    }

    if (n_splits_pp == n_splits_tg) {
        LLAMA_LOG_INFO("%s: graph splits = %d\n", __func__, n_splits_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph splits = %d (with bs=%d), %d (with bs=1)\n", __func__, n_splits_pp, n_tokens, n_splits_tg);
    }

    const int64_t t_end_us = ggml_time_us();

    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n",
            __func__, (t_end_us - t_start_us)/1000.0, ggml_backend_sched_get_n_copies(sched.get()));

    dense_graph_cache_begin_phase(dense_gpu_prefill_phase_active ? 1 : 2);
}

void llama_context::synchronize() {
    if (!sched) {
        return;
    }

    const bool graph_profile = kimi_graph_profile_enabled();
    const int  queued_tokens = n_queued_tokens;
    const uint64_t layer_gap_sync_start_us =
        g_kimi_layer_gap_context_enabled ? (uint64_t) ggml_time_us() : 0;
    const int64_t t_sync_start_us = graph_profile ? ggml_time_us() : 0;
    ggml_backend_sched_synchronize(sched.get());
    if (g_kimi_layer_gap_context_enabled) {
        kimi_layer_gap_context_thread_buffer().sync_entries.push_back({
            (uint32_t) std::max(0, queued_tokens),
            layer_gap_sync_start_us,
            (uint64_t) ggml_time_us(),
        });
    }
    if (graph_profile) {
        const uint64_t sync_us = (uint64_t) (ggml_time_us() - t_sync_start_us);
        g_kimi_graph_profile.sync_calls.fetch_add(1, std::memory_order_relaxed);
        g_kimi_graph_profile.sync_us.fetch_add(sync_us, std::memory_order_relaxed);
        if (queued_tokens == 1) {
            g_kimi_graph_profile.sync_decode_calls.fetch_add(1, std::memory_order_relaxed);
            g_kimi_graph_profile.sync_decode_us.fetch_add(sync_us, std::memory_order_relaxed);
            g_kimi_graph_profile.sync_decode_tokens.fetch_add(1, std::memory_order_relaxed);
        } else if (queued_tokens > 1) {
            g_kimi_graph_profile.sync_prompt_calls.fetch_add(1, std::memory_order_relaxed);
            g_kimi_graph_profile.sync_prompt_us.fetch_add(sync_us, std::memory_order_relaxed);
            g_kimi_graph_profile.sync_prompt_tokens.fetch_add((uint64_t) queued_tokens, std::memory_order_relaxed);
        } else {
            g_kimi_graph_profile.sync_idle_calls.fetch_add(1, std::memory_order_relaxed);
            g_kimi_graph_profile.sync_idle_us.fetch_add(sync_us, std::memory_order_relaxed);
        }
    }

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0;
    t_compute_start_us = 0;
}

bool llama_context::host_reserve(size_t bytes) {
    return !cpu_sink_runtime || cpu_sink_runtime->make_room(bytes);
}

bool llama_context::moe_prefill_phase_workspace(bool active) {
    return moe_prefill_phase_workspace(active, 0);
}

bool llama_context::moe_prefill_phase_workspace(bool active, uint32_t requested_tokens) {
    return moe_prefill_phase_workspace(active, requested_tokens, requested_tokens);
}

bool llama_context::moe_prefill_phase_workspace(bool active, uint32_t requested_tokens, uint32_t prefill_tokens) {
    const int64_t transition_start_us = ggml_time_us();
    auto complete_vram_phase = [&](bool decode) {
        return !cpu_sink_runtime || !cpu_sink_runtime->vram_budget || cpu_sink_runtime->phase_complete(decode);
    };
    // One line per prefill with the expert copies of its graph-route passes: what they read, how fast, and whether
    // a read fell back to a slower path.
    if (active || prefill_copy.passes > 0) {
        // the scheduler reads a split's experts before it launches the split: the passes' reads are all counted
        ggml_backend_sched_kimi_copy_stats now = {};
        ggml_backend_sched_kimi_copy_stats_get(&now);
        if (active) {
            prefill_copy.begin(now);
        } else {
            LLAMA_LOG_INFO("%s\n", prefill_copy.end(now).c_str());
        }
    }
    // One line per request phase with the copies of the weights homed on the drive: a request's start ends the
    // previous phase and begins its prefill, the prefill's end begins its decode, the request's end ends that
    {
        ggml_backend_sched_kimi_copy_stats now = {};
        ggml_backend_sched_kimi_copy_stats_get(&now);
        const bool decode_begins = !active && dense_stream.phase != nullptr &&
            std::strcmp(dense_stream.phase, "prefill") == 0;
        if (active || dense_stream.phase != nullptr) {
            const std::string line = dense_stream.end(now);
            if (!line.empty()) {
                LLAMA_LOG_INFO("%s\n", line.c_str());
            }
            dense_stream.phase = nullptr;
        }
        if (active || decode_begins) {
            dense_stream.begin(active ? "prefill" : "decode", now);
        }
    }
    // This is the explicit request prefill boundary, including single-token
    // prompts. Release before every early return or prefill workspace reserve.
    if (active && cpu_sink_runtime) {
        if (sched) { synchronize(); }
        cpu_sink_runtime->begin(nullptr, 0, false);
        cpu_sink_cache_batch_decode = false;
    }
    const bool graph_phase_gate_enabled = dense_gpu_phase_gate_enabled ||
        phase_decode_cache_handover_enabled;
    const bool cpu_sink_phase_changed = cpu_sink_enabled &&
        cpu_sink_decode_active != !active;
    const bool phase_changed = cpu_sink_phase_changed || (graph_phase_gate_enabled &&
        dense_gpu_prefill_phase_active != active);
    if (graph_phase_gate_enabled) {
        dense_gpu_prefill_phase_active = active;
    }
    if (cpu_sink_enabled) {
        cpu_sink_decode_active = !active;
        if (cpu_sink_phase_changed) {
            LLAMA_LOG_INFO("[kimi_cpu_sink_phase] decode_active=%d\n", cpu_sink_decode_active ? 1 : 0);
        }
    }

    auto drop_scheduler = [&]() {
        if (sched) {
            synchronize();
            dense_graph_cache_end_phase();
        }
        gf_res_prev.reset();
        gf_res_reserve.reset();
        sched.reset();
        sched_need_reserve = true;
    };
    // A prefill graph covers the whole prompt it was sized for; the single-token
    // decode graph covers what the prefill left in memory and grows in decode().
    const uint32_t prefill_horizon = !reserve_horizon_enabled ? 0 :
        requested_tokens ? (GGML_PAD(requested_tokens, 256u) < cparams.n_ctx ? GGML_PAD(requested_tokens, 256u) : 0) :
        reserve_horizon_bucket(memory_n_pos() + reserve_horizon_min);
    auto reserve_ubatch = [&](uint32_t ubatch) {
        cparams.n_ubatch = ubatch;
        drop_scheduler();
        if (cpu_prefill_output_sizing_enabled) {
            sched_reserve_n_outputs = cparams.n_seq_max;
        }
        sched_reserve_n_ctx = active && ubatch > 1 ? prefill_horizon :
            reserve_horizon_bucket(memory_n_pos() + 1);
        try {
            sched_reserve();
            return true;
        } catch (const std::exception &err) {
            LLAMA_LOG_ERROR(
                "%s: graph reserve failed for ubatch=%u: %s\n",
                __func__, ubatch, err.what());
        } catch (...) {
            LLAMA_LOG_ERROR(
                "%s: graph reserve failed for ubatch=%u with unknown error\n",
                __func__, ubatch);
        }
        drop_scheduler();
        return false;
    };

    if (cpu_prefill_actual_shape_enabled) {
        if (active) {
            if (cpu_prefill_actual_shape_active) {
                return complete_vram_phase(false);
            }
            const uint64_t requested = prefill_tokens ? prefill_tokens : cpu_prefill_configured_ubatch;
            const uint64_t rounded = ((requested + cpu_prefill_graph_granularity - 1) /
                cpu_prefill_graph_granularity) * cpu_prefill_graph_granularity;
            const uint32_t target = std::max<uint32_t>(1, std::min<uint64_t>(
                cpu_prefill_configured_ubatch, rounded));
            // A smaller graph when the request's own does not fit: the normal batch allocator splits the prompt into
            // physical ubatches, and each one streams its layers' expert union from the SSD again
            // (llama_prefill_ubatch_search).
            cpu_prefill_request_max_tokens = cpu_prefill_max_tokens;
            const uint32_t fallback = llama_prefill_ubatch_search(target, cpu_prefill_graph_granularity,
                cpu_prefill_request_max_tokens,
                [&](uint32_t n, uint32_t max_tokens) { return cpu_sink_prefill_route(n, max_tokens); },
                reserve_ubatch);
            if (fallback == target && cpu_prefill_request_max_tokens == cpu_prefill_max_tokens) {
                cpu_prefill_actual_shape_active = true;
                cpu_prefill_selected_ubatch = target;
                LLAMA_LOG_INFO(
                    "[kimi_cpu_prefill_graph] event=activate requested_tokens=%u selected_ubatch=%u "
                    "configured_max=%u granularity=%u transition_ms=%.3f\n",
                    prefill_tokens, target, cpu_prefill_configured_ubatch,
                    cpu_prefill_graph_granularity,
                    (ggml_time_us() - transition_start_us) / 1000.0);
                return complete_vram_phase(false);
            }
            if (fallback) {
                cpu_prefill_actual_shape_active = true;
                cpu_prefill_selected_ubatch = fallback;
                char max_tokens[16] = "all";
                if (cpu_prefill_request_max_tokens != std::numeric_limits<uint32_t>::max()) {
                    snprintf(max_tokens, sizeof(max_tokens), "%u", cpu_prefill_request_max_tokens);
                }
                LLAMA_LOG_WARN(
                    "[kimi_cpu_prefill_graph] event=fallback requested_tokens=%u target_ubatch=%u "
                    "selected_ubatch=%u route=%s max_sink_tokens=%s transition_ms=%.3f\n",
                    prefill_tokens, target, fallback, cpu_sink_prefill_route(fallback) ? "sink" : "graph",
                    max_tokens, (ggml_time_us() - transition_start_us) / 1000.0);
                return complete_vram_phase(false);
            }

            cpu_prefill_request_max_tokens = cpu_prefill_max_tokens;
            const bool decode_restored = reserve_ubatch(1);
            cpu_prefill_selected_ubatch = 0;
            complete_vram_phase(decode_restored);
            LLAMA_LOG_ERROR(
                "[kimi_cpu_prefill_graph] event=activate_failed requested_tokens=%u target_ubatch=%u "
                "fallback_ubatch=%u decode_restored=%d transition_ms=%.3f\n",
                prefill_tokens, target, fallback, decode_restored ? 1 : 0,
                (ggml_time_us() - transition_start_us) / 1000.0);
            return false;
        }

        if (!cpu_prefill_actual_shape_active) {
            const bool reserved = !phase_changed || reserve_ubatch(1);
            const bool vram_complete = complete_vram_phase(reserved);
            return reserved && vram_complete;
        }
        const uint32_t released = cpu_prefill_selected_ubatch;
        cpu_prefill_request_max_tokens = cpu_prefill_max_tokens;
        const bool graph_restored = reserve_ubatch(1);
        cpu_prefill_actual_shape_active = false;
        cpu_prefill_selected_ubatch = 0;
        const bool vram_complete = complete_vram_phase(graph_restored);
        LLAMA_LOG_INFO(
            "[kimi_cpu_prefill_graph] event=release selected_ubatch=%u decode_graph_restored=%d "
            "transition_ms=%.3f\n",
            released, graph_restored ? 1 : 0,
            (ggml_time_us() - transition_start_us) / 1000.0);
        return graph_restored && vram_complete;
    }

    if (!phase_workspace_enabled) {
        const bool reserved = !phase_changed || reserve_ubatch(cparams.n_ubatch);
        const bool vram_complete = complete_vram_phase(reserved && !active);
        return reserved && vram_complete;
    }
#if defined(_WIN32)
    complete_vram_phase(false);
    return false;
#else
    using workspace_begin_fn = bool (*)(uint32_t);
    using workspace_end_fn = bool (*)(void);
    auto * workspace_begin = reinterpret_cast<workspace_begin_fn>(
        dlsym(RTLD_DEFAULT, "ggml_cuda_moe_prefill_phase_workspace_begin"));
    auto * workspace_end = reinterpret_cast<workspace_end_fn>(
        dlsym(RTLD_DEFAULT, "ggml_cuda_moe_prefill_phase_workspace_end"));
    if (!workspace_begin || !workspace_end) {
        LLAMA_LOG_ERROR("%s: CUDA phase workspace API is unavailable\n", __func__);
        complete_vram_phase(false);
        return false;
    }

    if (active) {
        if (phase_workspace_active) {
            return complete_vram_phase(false);
        }
        // Dense staging belongs to decode.  Release it before the prefill
        // cache transaction measures free VRAM and chooses its phase slots.
        if (dense_gpu_phase_gate_enabled || cpu_sink_phase_changed) {
            drop_scheduler();
        } else {
            synchronize();
        }
        if (workspace_begin(phase_workspace_target_ubatch) &&
                reserve_ubatch(phase_workspace_target_ubatch)) {
            phase_workspace_active = true;
            phase_workspace_selected_ubatch = phase_workspace_target_ubatch;
            LLAMA_LOG_INFO(
                "%s: activated target ubatch=%u transition_ms=%.3f\n",
                __func__, phase_workspace_selected_ubatch,
                (ggml_time_us() - transition_start_us) / 1000.0);
            return complete_vram_phase(false);
        }

        // Undo every target-level allocation before attempting the previous
        // validated capacity level with the full decode cache geometry.
        drop_scheduler();
        workspace_end();
        const uint32_t fallback_ubatch = phase_workspace_target_ubatch > 64 ? 64 :
            phase_workspace_target_ubatch > 32 ? 32 : 16;
        if (workspace_begin(fallback_ubatch) && reserve_ubatch(fallback_ubatch)) {
            phase_workspace_active = true;
            phase_workspace_selected_ubatch = fallback_ubatch;
            LLAMA_LOG_WARN(
                "%s: target ubatch=%u unavailable; safely fell back to ubatch=%u "
                "transition_ms=%.3f\n",
                __func__, phase_workspace_target_ubatch, fallback_ubatch,
                (ggml_time_us() - transition_start_us) / 1000.0);
            return complete_vram_phase(false);
        }

        drop_scheduler();
        workspace_end();
        if (graph_phase_gate_enabled) {
            dense_gpu_prefill_phase_active = false;
        }
        const bool decode_restored = reserve_ubatch(1);
        phase_workspace_selected_ubatch = 0;
        LLAMA_LOG_ERROR(
            "%s: target and fallback prefill reserves failed; decode graph restored=%d "
            "transition_ms=%.3f\n",
            __func__, decode_restored ? 1 : 0,
            (ggml_time_us() - transition_start_us) / 1000.0);
        complete_vram_phase(decode_restored);
        return false;
    }

    if (!phase_workspace_active) {
        const bool reserved = !phase_changed || reserve_ubatch(1);
        const bool vram_complete = complete_vram_phase(reserved);
        return reserved && vram_complete;
    }
    // The large graph must die before the cache pools are restored, otherwise
    // the two phases overlap and recreate the 3.1 GiB cliff.
    drop_scheduler();
    const bool cache_restored = workspace_end();
    const bool graph_restored = reserve_ubatch(1);
    LLAMA_LOG_INFO(
        "%s: released ubatch=%u cache_restored=%d decode_graph_restored=%d "
        "transition_ms=%.3f\n",
        __func__, phase_workspace_selected_ubatch,
        cache_restored ? 1 : 0,
        graph_restored ? 1 : 0,
        (ggml_time_us() - transition_start_us) / 1000.0);
    phase_workspace_active = false;
    phase_workspace_selected_ubatch = 0;
    const bool vram_complete = complete_vram_phase(cache_restored && graph_restored);
    return cache_restored && graph_restored && vram_complete;
#endif
}

const llama_k3_runtime_config * llama_context::k3_runtime_config_get() const {
    return k3_runtime.get();
}

bool llama_context::k3_runtime_guard_set(bool active) {
    if (!k3_runtime || (active && k3_runtime->guard_tokens == 0 && k3_runtime->guard_start_tokens == 0 &&
            k3_runtime->guard_control_tokens == 0)) { return false; }
    if (active == k3_guard_active) { return true; }
    synchronize();
    k3_guard_active = active;
    sched_need_reserve = true;
    return true;
}

const llama_k3_runtime_config * llama_k3_runtime_config_get(const llama_context * ctx) {
    return ctx ? ctx->k3_runtime_config_get() : nullptr;
}

bool llama_k3_runtime_guard_set(llama_context * ctx, bool active) {
    return ctx && ctx->k3_runtime_guard_set(active);
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched.get();
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    return memory.get();
}

bool llama_context::memory_update(bool optimize) {
    if (!memory) {
        return false;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return false;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return false;
                }
        }

        // reset the previous graph result to make sure that it won't be reused
        // TODO: change the mctx->apply() to return information if a graph reserve is needed
        //       reset the graph result only if the memory module did reset the scheduler
        gf_res_prev->reset();

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const auto mctx = memory_init_reserve();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory context");
        }

        const uint32_t n_seqs = cparams.n_seq_max;
        const uint32_t n_tokens = kimi_cpu_sink_reserve_tokens(cparams,
            cpu_sink_enabled && cpu_sink_decode_active && !cparams.warmup);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_tokens, mctx.get());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
        }
    }

    return true;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    output_reorder();

    return logits.data;
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * llama_context::get_logits_ith(int32_t i) {
    output_reorder();

    try {
        if (logits.data == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);
        float * result = logits.data + j*model.vocab.n_tokens();
        kimi_logits_trace_write(result, model.vocab.n_tokens(), i);
        return result;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd.data;
}

llama_token * llama_context::get_sampled_tokens()  const{
    return sampling.sampled.data;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    output_reorder();

    try {
        if (embd.data == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        const int64_t j = output_resolve_row(i);
        const uint32_t n_embd_out = model.hparams.n_embd_out();
        return embd.data + j*n_embd_out;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

float * llama_context::get_embeddings_nextn() {
    output_reorder();
    return embd_nextn.data;
}

float * llama_context::get_embeddings_nextn_ith(int32_t i) {
    output_reorder();

    try {
        if (!embd_nextn.has_data()) {
            throw std::runtime_error("no nextn embeddings");
        }
        const int64_t row = cparams.embeddings_nextn_masked ? output_resolve_row(i) :
            (i < 0 ? (int64_t) n_outputs + i : i);
        if (row < 0 || (size_t) row * model.hparams.n_embd >= embd_nextn.size) {
            throw std::runtime_error("nextn embedding index out of range");
        }
        return embd_nextn.data + row * model.hparams.n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid nextn embeddings id %d, reason: %s\n", __func__, i, err.what());
        return nullptr;
    }
}

float * llama_context::get_embeddings_layer_inp(uint32_t lid) {
    GGML_ASSERT(lid < embd_layer_inp.size() && embd_layer_inp[lid].has_data());
    return embd_layer_inp[lid].data;
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
        return LLAMA_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        GGML_ASSERT(row < (int64_t) sampling.sampled.size);
        return sampling.sampled.data[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        return LLAMA_TOKEN_NULL;
    }
}

float * llama_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        return sampling.probs.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

float * llama_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        return sampling.logits.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (sampling.candidates.has_data() &&
            (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            return sampling.candidates.data + row*model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
        GGML_UNUSED(err);
    }

    return sampling.token_ids_full_vocab.data();
}

size_t llama_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (!sampling.candidates.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}


void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        if (reg) {
            auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
            if (set_abort_callback_fn) {
                set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
            }
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cpu_prefill_output_sizing_enabled && cparams.embeddings != value) {
        if (sched) {
            synchronize();
            dense_graph_cache_end_phase();
        }
        gf_res_prev.reset();
        gf_res_reserve.reset();
        sched.reset();
        sched_need_reserve = true;
        if (!value) {
            sched_reserve_n_outputs = cparams.n_seq_max;
        }
    }
    cparams.embeddings = value;
}

void llama_context::set_embeddings_nextn(bool value, bool masked) {
    LLAMA_LOG_DEBUG("%s: value = %d, masked = %d\n", __func__, value, masked);

    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;
    sched_need_reserve = true;
}

void llama_context::set_embeddings_layer_inp(uint32_t lid, bool enable) {
    LLAMA_LOG_DEBUG("%s: lid = %u, enable = %d\n", __func__, lid, enable);

    GGML_ASSERT(lid < cparams.embeddings_layer_inp.size());
    cparams.embeddings_layer_inp[lid] = enable;
    sched_need_reserve = true;
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.causal_attn == value) {
        return;
    }

    cparams.causal_attn = value;

    sched_need_reserve = true;
}

void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.warmup == value) {
        return;
    }

    cparams.warmup = value;

    // warmups are usually with small batches, so no need to reserve
    //sched_need_reserve = true;
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    const bool can_offload =
        sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        llama_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());

        sampler->iface->backend_init(sampler, buft);

        sampling.samplers[seq_id] = sampler;

        sched_need_reserve = true;

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__, llama_sampler_name(sampler), seq_id);

        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }

        sampling.samplers.erase(seq_id);

        return false;
    }

    sampling.samplers.erase(seq_id);

    sched_need_reserve = true;

    return true;
}

void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    if (adapters_lora_are_same(adapters, n_adapters, scales)) {
        return;
    }

    loras.reset(new llama_adapter_loras());

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] != 0.0f) {
            loras->insert({adapters[i], scales[i]});
        }
    }

    sched_need_reserve = true;
}

bool llama_context::adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
    size_t n_non_zero = 0;

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] == 0.0f) {
            continue;
        }
        n_non_zero++;

        auto it = loras->find(adapters[i]);

        if (it == loras->end() || it->second != scales[i]) {
            return false;
        }
    }

    if (n_non_zero != loras->size()) {
        return false;
    }

    return true;
}

bool llama_context::set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    bool res = cvec->apply(model, data, len, n_embd, il_start, il_end);

    sched_need_reserve = true;

    return res;
}

llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    if (cpu_sink_decode_active && ubatch.n_tokens > 1 && !cpu_sink_multi_token_reported) {
        // the sink decodes one token at a time and the prefill route is for prefill: a larger decode ubatch (a draft
        // to verify) runs ggml's MUL_MAT_ID where the scheduler puts it
        LLAMA_LOG_WARN("[kimi_cpu_sink] n_tokens=%u in decode runs ggml's MUL_MAT_ID %s; the sink decodes one token\n",
            ubatch.n_tokens, gpu_offloads_experts(ubatch.n_tokens) ? "on the GPU" : "on the CPU over the mapped experts");
        cpu_sink_multi_token_reported = true;
    }
    kimi_layer_gap_context_entry layer_gap_entry;
    if (g_kimi_layer_gap_context_enabled) {
        layer_gap_entry.call_index =
            g_kimi_layer_gap_context_call_index.fetch_add(1, std::memory_order_relaxed);
        layer_gap_entry.n_tokens = ubatch.n_tokens;
        layer_gap_entry.call_start_us = (uint64_t) ggml_time_us();
    }
    if (mctx && !mctx->apply()) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    ggml_backend_sched_kimi_set_dense_decode(
        sched.get(),
        !dense_gpu_phase_gate_enabled ||
            (!dense_gpu_prefill_phase_active && ubatch.n_seq_tokens == 1));

    auto * res = gf_res_prev.get();
    auto * gf  = res->get_gf();

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
    if (g_kimi_layer_gap_context_enabled) {
        layer_gap_entry.graph_prepare_start_us = (uint64_t) ggml_time_us();
    }
    const auto gparams = graph_params(res, ubatch, mctx, gtype);
    if (dense_stream.phase != nullptr) {
        dense_stream.add(ubatch.n_tokens);
    }
    if (cpu_sink_enabled && !cpu_sink_decode_active && !cparams.warmup) {
        char max_tokens[16] = "all";
        if (cpu_prefill_request_max_tokens != std::numeric_limits<uint32_t>::max()) {
            snprintf(max_tokens, sizeof(max_tokens), "%u", cpu_prefill_request_max_tokens);
        }
        LLAMA_LOG_INFO("[kimi_prefill_route] n_tokens=%u route=%s max_sink_tokens=%s offloads=%d\n",
            ubatch.n_tokens, gparams.cpu_sink_prefill ? "sink" : "graph", max_tokens,
            gpu_offloads_experts(ubatch.n_tokens) ? 1 : 0);
        if (!gparams.cpu_sink_prefill) {
            prefill_copy.pass(ubatch.n_tokens);
        }
    }

    if (!graph_reuse_disable && res->can_reuse(gparams)) {
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\n", __func__);

        // with pipeline parallelism, the previous graph_compute_async may still be running
        // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
        // that the previous compute is still reading.
        if (cparams.pipeline_parallel) {
            ggml_backend_sched_synchronize(sched.get());
        }

        n_reused++;
        if (g_kimi_layer_gap_context_enabled) {
            layer_gap_entry.reused = 1;
        }
    } else {
        res->reset();

        ggml_backend_sched_reset(sched.get());
        if (moe_next_gate_prefetch_probe_out_path()) {
            ggml_backend_sched_set_eval_callback(sched.get(), moe_next_gate_prefetch_probe_cb_eval, this);
        } else {
            ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);
        }

        //const auto t_start_us = ggml_time_us();

        if (g_kimi_layer_gap_context_enabled) {
            layer_gap_entry.graph_build_start_us = (uint64_t) ggml_time_us();
        }
        gf = model.build_graph(gparams);
        if (g_kimi_layer_gap_context_enabled) {
            layer_gap_entry.graph_build_end_us = (uint64_t) ggml_time_us();
        }

        //LLAMA_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }
        if (g_kimi_layer_gap_context_enabled) {
            layer_gap_entry.graph_alloc_end_us = (uint64_t) ggml_time_us();
        }
    }

    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        if (g_kimi_layer_gap_context_enabled) {
            layer_gap_entry.set_inputs_start_us = (uint64_t) ggml_time_us();
        }
        res->set_inputs(&ubatch);
        if (g_kimi_layer_gap_context_enabled) {
            layer_gap_entry.set_inputs_end_us = (uint64_t) ggml_time_us();
        }

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
    }

    if (g_kimi_layer_gap_context_enabled) {
        layer_gap_entry.compute_start_us = (uint64_t) ggml_time_us();
    }
    // Phase/token tracking is also required by a VRAM-only tier and by
    // zero-cache profiling. The runtime skips this when all are disabled.
    if (cpu_sink_cache_batch_decode && cpu_sink_runtime) {
        // One begin per actual logical decode token, on new and reused graphs.
        // Whole-batch eligibility excludes one-token tails of a larger batch.
        ggml_backend_sched_synchronize(sched.get());
        const bool cache_decode = cpu_sink_cache_batch_decode &&
            gtype == LLM_GRAPH_TYPE_DECODER && !res->cpu_sink_nodes.empty() &&
            ubatch.n_tokens == 1 && ubatch.n_seqs_unq == 1 &&
            ubatch.n_seq_id[0] == 1 && ubatch.pos[0] >= 0;
        auto * registry = backend_cpu ? ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu)) : nullptr;
        cpu_sink_runtime->begin(registry, cache_decode ? static_cast<uint64_t>(ubatch.pos[0]) : 0,
            cache_decode, cache_decode ? ubatch.seq_id[0][0] : 0);
    }
    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (g_kimi_layer_gap_context_enabled) {
        layer_gap_entry.compute_end_us = (uint64_t) ggml_time_us();
    }
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        return nullptr;
    }

    moe_next_gate_shadow_record(res, ubatch, sched.get());
    moe_route_score_trace_record(res, ubatch, sched.get());

    ret = GGML_STATUS_SUCCESS;
    if (g_kimi_layer_gap_context_enabled) {
        layer_gap_entry.call_end_us = (uint64_t) ggml_time_us();
        kimi_layer_gap_context_thread_buffer().entries.push_back(layer_gap_entry);
    }

    return res;
}

int llama_context::encode(const llama_batch & batch_inp) {
    GGML_ASSERT((!batch_inp.token && batch_inp.embd) || (batch_inp.token && !batch_inp.embd)); // NOLINT

    cpu_sink_cache_batch_decode = false;
    if (cpu_sink_runtime) {
        if (sched) { synchronize(); }
        cpu_sink_runtime->begin(nullptr, 0, false);
    }

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    const int64_t n_embd  = hparams.n_embd_inp_enc();
    const int64_t n_vocab = model.vocab.n_tokens();

    // note: during encode, we always pass the full sequence starting from pos = 0
    if (!balloc->init(batch_inp, model.vocab, nullptr, n_embd, cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    // TODO: this clear of the buffer can easily be forgotten - need something better
    embd_seq.clear();

    sched_reserve();

    n_queued_tokens += n_tokens;

    // reserve output buffer
    if (output_reserve(n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (uint32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }

    n_outputs = n_tokens;

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status status;
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, status);

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:      return  2;
            case GGML_STATUS_ALLOC_FAILED: return -2;
            case GGML_STATUS_FAILED:       return -3;
            case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
        }
    }

    auto * t_logits = res->get_logits();
    auto * t_embd = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();
    auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;

    // extract logits
    if (logits.data && t_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits.data != nullptr);

        ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, n_tokens*n_vocab*sizeof(float));
    }

    // extract embeddings
    if (embd.data && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd.data != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out();

                    GGML_ASSERT(n_tokens*n_embd_out <= (int64_t) embd.size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd.data, 0, n_tokens*n_embd_out*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        embd_seq_out[seq_id].resize(n_embd_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    if (embd_nextn.data && t_h_nextn) {
        ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
        GGML_ASSERT(backend_h != nullptr);
        GGML_ASSERT((size_t) n_tokens * hparams.n_embd <= embd_nextn.size);
        ggml_backend_tensor_get_async(
            backend_h, t_h_nextn, embd_nextn.data, 0,
            (size_t) n_tokens * hparams.n_embd * sizeof(float));
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd.data, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

static std::map<llama_seq_id, uint32_t> build_seq_to_output_row(const llama_ubatch & ubatch, uint32_t row_offset) {
    std::map<llama_seq_id, uint32_t> seq_to_row;
    // how many output tokens we have seen so far for this ubatch.
    uint32_t local = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        // skip tokens that are not output.
        if (!ubatch.output[i]) {
            continue;
        }

        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        // row_offset is the number of output tokens before this ubatch.
        seq_to_row[seq_id] = row_offset + local;
        ++local;
    }
    return seq_to_row;
}

static void copy_tensor_async_ints(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<llama_token> & sampled,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!sampled.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < sampled.size);

        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampled tokens tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        ggml_backend_tensor_get_async(backend, tensor, sampled.data + row, 0, sizeof(sampled.data[row]));
    }
}

static void copy_tensor_async_floats(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<float> & dst,
    size_t stride,
    std::vector<uint32_t> & counts,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!dst.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < counts.size());

        GGML_ASSERT(ggml_is_contiguous(tensor) && "logits/probs tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        float * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        // Update the actual number of logits/probabilities that were written for this row.
        counts[row] = ggml_nelements(tensor);
    }
}

static void copy_tensor_async_candidates(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<llama_token> & dst,
    size_t stride,
    std::vector<uint32_t> & counts,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!dst.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < counts.size());

        GGML_ASSERT(ggml_is_contiguous(tensor) && "candidates tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        llama_token * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        // Update the actual number of candidates that were written.
        counts[row] = ggml_nelements(tensor);
    }
}

static bool needs_raw_logits(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) {
            continue;
        }

        // Check if the output token has at least one sequence without a backend sampler.
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) {
                return true;
            }
        }
    }
    return false; // all sequences use backend sampling
}

int llama_context::decode(const llama_batch & batch_inp) {
    GGML_ASSERT((!batch_inp.token && batch_inp.embd) || (batch_inp.token && !batch_inp.embd)); // NOLINT

    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    char nvtx_decode_label[96];
    std::snprintf(nvtx_decode_label, sizeof(nvtx_decode_label), "%s tokens=%d",
            batch_inp.n_tokens > 1 ? "prompt_phase" : "decode_token",
            batch_inp.n_tokens);
    llama_nvtx_range nvtx_decode_scope(nvtx_decode_label);

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int64_t n_vocab = vocab.n_tokens();
    const int64_t n_embd  = hparams.n_embd_inp();

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    // TODO: avoid this workaround in the future
    if (has_samplers && batch_inp.logits) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (int32_t i = 0; i < batch_inp.n_tokens; ++i) {
            if (batch_inp.logits[i] == 0) {
                continue;
            }

            const int ns = batch_inp.n_seq_id ? batch_inp.n_seq_id[i] : 1;

            for (int32_t s = 0; s < ns; ++s) {
                const llama_seq_id seq_id = batch_inp.seq_id ? batch_inp.seq_id[i][s] : 0;

                seq_output_count[seq_id]++;
                if (seq_output_count[seq_id] > 1) {
                    LLAMA_LOG_ERROR("%s: backend sampling requires at most one output token per sequence (seq_id %d had %d)\n",
                            __func__, seq_id, seq_output_count[seq_id]);
                    return -1;
                }
            }
        }
    }

    if (!balloc->init(batch_inp, vocab, memory.get(), n_embd, n_seq_max, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    cpu_sink_cache_batch_decode = cpu_sink_enabled && cpu_sink_decode_active &&
        !cparams.warmup && n_tokens_all == 1;
    const bool cpu_sink_non_decode_phase = !cpu_sink_cache_batch_decode && cpu_sink_runtime &&
        (cpu_sink_runtime->cache_budget || cpu_sink_runtime->vram_budget);

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    // TODO: this clear of the buffer can easily be forgotten - need something better
    embd_seq.clear();
    output_swaps.clear();

    auto reserve_with_tier = [&](bool tier_phase, bool decode_phase) {
        if (tier_phase) {
            if (sched) { synchronize(); }
            cpu_sink_runtime->begin(nullptr, 0, false);
        }
        sched_reserve();
        return !tier_phase || cpu_sink_runtime->phase_complete(decode_phase);
    };
    auto reset_scheduler = [&]() {
        if (sched) {
            synchronize();
            dense_graph_cache_end_phase();
        }
        gf_res_prev.reset();
        gf_res_reserve.reset();
        sched.reset();
        sched_need_reserve = true;
    };

    bool reserve_grown = false;
    if (sched_reserve_n_ctx != 0) {
        // the reserved graph covers sched_reserve_n_ctx positions; past that, reserve the next bucket
        uint32_t n_pos = memory_n_pos();
        const llama_batch & batch_all = balloc->get_batch();
        for (int32_t i = 0; i < batch_all.n_tokens; ++i) {
            n_pos = std::max<uint32_t>(n_pos, (uint32_t) batch_all.pos[i] + 1);
        }
        if (n_pos > sched_reserve_n_ctx) {
            const uint32_t prev = sched_reserve_n_ctx;
            reset_scheduler();
            sched_reserve_n_ctx = reserve_horizon_bucket(n_pos);
            reserve_grown = true;
            LLAMA_LOG_INFO("[reserve_horizon] grow n_pos=%u from=%u to=%u\n",
                n_pos, prev, sched_reserve_n_ctx ? sched_reserve_n_ctx : cparams.n_ctx);
        }
    }
    // A decode step that outgrows the reserve takes the larger buffers the way a prefill does:
    // the VRAM tier yields blocks on demand while they are allocated and grows back afterwards.
    const bool reserve_grow_phase = (reserve_grown || sched_need_reserve) && cpu_sink_cache_batch_decode && cpu_sink_runtime &&
        (cpu_sink_runtime->cache_budget || cpu_sink_runtime->vram_budget);
    if (!reserve_with_tier(cpu_sink_non_decode_phase || reserve_grow_phase, reserve_grow_phase)) {
        LLAMA_LOG_ERROR("%s: VRAM tier phase completion failed after scheduler reserve\n", __func__);
        return -1;
    }

    bool did_optimize = false;

    // handle any pending shifts/copies
    memory_update(false);

    llama_memory_context_ptr mctx;

    while (true) {
        mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                {
                    if (!did_optimize) {
                        did_optimize = true;

                        if (memory_update(true)) {
                            LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, balloc->get_n_tokens());

                            continue;
                        }
                    }

                    LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, balloc->get_n_tokens());

                    return 1;
                }
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    int64_t n_outputs_prev = 0;
    int64_t n_tokens_prev  = 0;

    do {
        const auto & ubatch = mctx->get_ubatch();

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        if (cpu_prefill_output_sizing_enabled && n_outputs > 0 &&
                static_cast<uint32_t>(n_outputs) > sched_reserve_n_outputs) {
            const uint32_t previous = sched_reserve_n_outputs;
            reset_scheduler();
            sched_reserve_n_outputs = static_cast<uint32_t>(n_outputs);
            LLAMA_LOG_INFO("[prefill_reserve_outputs] event=grow from=%u to=%u ubatch_tokens=%u\n",
                previous, sched_reserve_n_outputs, ubatch.n_tokens);

            const bool tier_phase = cpu_sink_runtime &&
                (cpu_sink_runtime->cache_budget || cpu_sink_runtime->vram_budget);
            if (!reserve_with_tier(tier_phase, cpu_sink_cache_batch_decode)) {
                LLAMA_LOG_ERROR("%s: VRAM tier phase completion failed after output reserve grew\n", __func__);
                return -1;
            }
        }

        ggml_status status;
        const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_DECODER, mctx.get(), status);

        if (!res) {
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                    continue;
                }

                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, pos_min[s]);

                memory->seq_rm(s, pos_min[s], -1);
            }

            switch (status) {
                case GGML_STATUS_ABORTED:      return  2;
                case GGML_STATUS_ALLOC_FAILED: return -2;
                case GGML_STATUS_FAILED:       return -3;
                case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        auto * t_logits = res->get_logits();
        auto * t_embd   = cparams.embeddings ? res->get_embd() : nullptr;
        auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;

        extract_layer_inputs(res, n_tokens_prev, ubatch.n_tokens);

        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // extract logits
        if (logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits.data != nullptr);

            float * logits_out = logits.data + n_outputs_prev*n_vocab;

            if (n_outputs) {
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }

        // extract embeddings
        if (embd.data && t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd.data != nullptr);
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        float * embd_out = embd.data + n_outputs_prev*n_embd_out;

                        if (n_outputs) {
                            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd_out <= (int64_t) embd.size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_embd_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - n_cls_out floats per sequence
                        auto & embd_seq_out = embd_seq;

                        const uint32_t n_cls_out = hparams.n_cls_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_cls_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        if (embd_nextn.data && t_h_nextn) {
            const size_t n_rows = cparams.embeddings_nextn_masked ? n_outputs : ubatch.n_tokens;
            const size_t row_offset = cparams.embeddings_nextn_masked ? n_outputs_prev : n_tokens_prev;
            if (n_rows > 0) {
                GGML_ASSERT((row_offset + n_rows) * hparams.n_embd <= embd_nextn.size);
                ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
                GGML_ASSERT(backend_h != nullptr);
                ggml_backend_tensor_get_async(
                    backend_h, t_h_nextn, embd_nextn.data + row_offset * hparams.n_embd,
                    0, n_rows * hparams.n_embd * sizeof(float));
            }
        }

        // Copy backend sampling output if this ubatch produced any sampling tensors.
        if (has_samplers && (!res->t_sampled.empty() || !res->t_sampled_probs.empty() || !res->t_sampled_logits.empty())) {
            const auto seq_to_output_row = build_seq_to_output_row(ubatch, n_outputs_prev);
            const auto stride = n_vocab;

            // async copy the sampling data from the backend to the host
            copy_tensor_async_ints(res->t_sampled, sampling.sampled, seq_to_output_row, sched.get());

            copy_tensor_async_floats    (res->t_sampled_logits, sampling.logits,     stride, sampling.logits_count,     seq_to_output_row, sched.get());
            copy_tensor_async_floats    (res->t_sampled_probs,  sampling.probs,      stride, sampling.probs_count,      seq_to_output_row, sched.get());
            copy_tensor_async_candidates(res->t_candidates,     sampling.candidates, stride, sampling.candidates_count, seq_to_output_row, sched.get());
        }

        n_outputs_prev += n_outputs;
        n_tokens_prev  += ubatch.n_tokens;
    } while (mctx->next());

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    const char * phase0_env = std::getenv("GGML_MOE_PREFILL_PHASE0");
    const char * phase1_env = std::getenv("GGML_MOE_PREFILL_PHASE1");
    const bool phase0_eviction = phase0_env && phase0_env[0] && phase0_env[0] != '0';
#if !defined(_WIN32)
    if (batch_inp.n_tokens > 1 && phase1_env && phase1_env[0] && phase1_env[0] != '0') {
        using moe_phase01_release_fn = void (*)(void);
        auto * moe_phase01_release = reinterpret_cast<moe_phase01_release_fn>(
            dlsym(RTLD_DEFAULT, "ggml_cuda_moe_prefill_phase01_release"));
        if (moe_phase01_release) {
            moe_phase01_release();
        }
    }
#endif
    if (batch_inp.n_tokens > 1 && (!cparams.warmup || !phase0_eviction)) {
        model.drop_expert_mmap_pages_after_prompt();
    }

    return 0;
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd     = hparams.n_embd;
    const auto n_embd_out = hparams.n_embd_out();

    bool has_logits = true;
    bool has_embd   = cparams.embeddings;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }


    size_t backend_float_count = 0;
    size_t backend_token_count = 0;
    size_t embd_layer_inp_float_count = 0;

    logits.size = has_logits ? n_vocab*n_outputs_max : 0;
    embd.size   = has_embd ? n_embd_out*n_outputs_max : 0;
    embd_nextn.size = cparams.embeddings_nextn ? (size_t) n_embd * n_batch : 0;

    for (bool enabled : cparams.embeddings_layer_inp) {
        if (enabled) {
            embd_layer_inp_float_count += (size_t) n_embd * n_batch;
        }
    }

    // Allocate backend sampling output buffers if there are backend samplers configured.
    const bool has_sampling = !sampling.samplers.empty();
    if (has_sampling) {
        backend_float_count = 2 * n_vocab * n_outputs_max;      // logits + probs
        backend_token_count = (1 + n_vocab) * n_outputs_max;    // sampled + candidates
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t new_size  =
        (logits.size + embd.size + embd_nextn.size + embd_layer_inp_float_count + backend_float_count) * sizeof(float) +
        (                                                                        backend_token_count) * sizeof(llama_token);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            synchronize();

            // TODO: not needed?
            buf_output = nullptr;
            logits.data = nullptr;
            embd.data = nullptr;
            embd_nextn.data = nullptr;
            for (auto & layer_inp : embd_layer_inp) {
                layer_inp = {nullptr, 0};
            }
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
        ggml_backend_buffer_clear(buf_output.get(), 0);
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    size_t offset = 0;
    uint8_t * base = (uint8_t *) output_base;

    logits = has_logits ? buffer_view<float>{output_base, logits.size} : buffer_view<float>{nullptr, 0};
    offset += logits.size * sizeof(float);

    embd = has_embd ? buffer_view<float>{(float *) (base + offset), embd.size} : buffer_view<float>{nullptr, 0};
    offset += embd.size * sizeof(float);

    embd_nextn = cparams.embeddings_nextn ?
        buffer_view<float>{(float *) (base + offset), embd_nextn.size} : buffer_view<float>{nullptr, 0};
    offset += embd_nextn.size * sizeof(float);

    for (uint32_t il = 0; il < embd_layer_inp.size(); ++il) {
        if (cparams.embeddings_layer_inp[il]) {
            embd_layer_inp[il] = buffer_view<float>{
                (float *) (base + offset), (size_t) n_embd * n_batch};
            offset += embd_layer_inp[il].size * sizeof(float);
        } else {
            embd_layer_inp[il] = {nullptr, 0};
        }
    }

    if (has_sampling) {
        sampling.logits = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.logits.size * sizeof(float);

        sampling.probs = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.probs.size * sizeof(float);

        sampling.sampled = {(llama_token *) (base + offset), (size_t)n_outputs_max};
        offset += sampling.sampled.size * sizeof(llama_token);

        sampling.candidates = {(llama_token *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.candidates.size * sizeof(llama_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(),     sampling.logits_count.end(),     0);
        std::fill(sampling.probs_count.begin(),      sampling.probs_count.end(),      0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
    } else {
        sampling.logits     = {nullptr, 0};
        sampling.probs      = {nullptr, 0};
        sampling.sampled    = {nullptr, 0};
        sampling.candidates = {nullptr, 0};

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;

    return n_outputs_max;
}

void llama_context::extract_layer_inputs(
        const llm_graph_result * res, size_t token_offset, size_t n_tokens) {
    for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
        }

        GGML_ASSERT(embd_layer_inp[il].has_data());
        ggml_tensor * tensor = res->get_layer_inp((int) il);
        GGML_ASSERT(tensor != nullptr);
        GGML_ASSERT(n_tokens > 0 && ggml_nbytes(tensor) % n_tokens == 0);

        const size_t row_bytes = ggml_nbytes(tensor) / n_tokens;
        const size_t dst_offset = token_offset * row_bytes;
        GGML_ASSERT(dst_offset + ggml_nbytes(tensor) <= embd_layer_inp[il].size * sizeof(float));

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), tensor);
        GGML_ASSERT(backend != nullptr);
        ggml_backend_tensor_get_async(
            backend, tensor, (uint8_t *) embd_layer_inp[il].data + dst_offset, 0, ggml_nbytes(tensor));
    }
}

void llama_context::output_reorder() {
    const uint64_t n_vocab = model.vocab.n_tokens();
    const uint64_t n_embd  = model.hparams.n_embd;

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits.size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits.data[i0*n_vocab + k], logits.data[i1*n_vocab + k]);
            }
        }

        if (embd.size > 0) {
            for (uint64_t k = 0; k < n_embd; k++) {
                std::swap(embd.data[i0*n_embd + k], embd.data[i1*n_embd + k]);
            }
        }

        if (embd_nextn.size > 0 && cparams.embeddings_nextn_masked) {
            for (uint64_t k = 0; k < n_embd; ++k) {
                std::swap(embd_nextn.data[i0*n_embd + k], embd_nextn.data[i1*n_embd + k]);
            }
        }

        for (auto & layer_inp : embd_layer_inp) {
            if (layer_inp.size > 0) {
                for (uint64_t k = 0; k < n_embd; ++k) {
                    std::swap(layer_inp.data[i0*n_embd + k], layer_inp.data[i1*n_embd + k]);
                }
            }
        }

        if (!sampling.samplers.empty()) {
            assert(sampling.logits.size > 0);
            assert(sampling.probs.size > 0);
            assert(sampling.candidates.size > 0);
            assert(sampling.sampled.size > 0);
            assert(sampling.logits_count.size() > 0);
            assert(sampling.probs_count.size() > 0);
            assert(sampling.candidates_count.size() > 0);

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.logits.data[i0*n_vocab + k], sampling.logits.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.probs.data[i0*n_vocab + k], sampling.probs.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.candidates.data[i0*n_vocab + k], sampling.candidates.data[i1*n_vocab + k]);
            }

            std::swap(sampling.sampled.data[i0],     sampling.sampled.data[i1]);
            std::swap(sampling.logits_count[i0],     sampling.logits_count[i1]);
            std::swap(sampling.probs_count[i0],      sampling.probs_count[i1]);
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        }
    }

    output_swaps.clear();
}

//
// graph
//

uint32_t llama_context::graph_max_nodes(uint32_t n_tokens) const {
    if (model.arch == LLM_ARCH_QWEN3NEXT || model.arch == LLM_ARCH_KIMI_LINEAR || model.arch == LLM_ARCH_KIMI_K3 ||
        model.arch == LLM_ARCH_QWEN35 || model.arch == LLM_ARCH_QWEN35MOE) {
        return std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors());
    }
    if (model.arch == LLM_ARCH_DEEPSEEK4) {
        return std::max<uint32_t>(n_tokens * 512, 256u * model.n_tensors());
    }
    if (model.arch == LLM_ARCH_DEEPSEEK41) {
        // same bound upstream uses for DeepSeek-V4.1
        return std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors());
    }
    uint32_t res = std::max<uint32_t>(1024u, 8u*model.n_tensors());
    for (const auto & lora : model.loras) {
        res += lora->get_n_nodes();
    }
    return res;
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

ggml_cgraph * llama_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only, size_t * sizes) {
    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1);

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs; // round to next multiple of n_seqs
        n_outputs = std::max(n_outputs, n_tokens);

        LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__, n_tokens, n_seqs, n_outputs);
    }

    const bool reserve_tg = n_tokens == n_seqs;
    ggml_backend_sched_kimi_set_dense_decode(
        sched.get(),
        !dense_gpu_phase_gate_enabled ||
            (!dense_gpu_prefill_phase_active && reserve_tg));
    // A phase-1 reserve constructs both the actual prompt graph and a synthetic
    // token-generation graph. Keep dense streaming phase-gated, but size and
    // split the synthetic tg graph with the KDA decode topology that will run
    // after the prompt. Real prompt execution still uses process_ubatch's
    // prefill assignment.
    ggml_backend_sched_kimi_set_kda_decode(
        sched.get(), !dense_gpu_phase_gate_enabled || reserve_tg);
    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannot reuse the old graph, so we reset the previous graph result to prevent that
    gf_res_prev->reset();

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs);

    // set one output token per sequence in order to activate all backend samplers
    std::vector<llama_seq_id> seq_ids(n_seqs);
    for (uint32_t i = 0; i < n_seqs; ++i) {
        seq_ids[i] = i;
        ubatch.n_seq_id[i] = 1;
        ubatch.seq_id[i] = &seq_ids[i];
        ubatch.output[i] = true;
    }

    auto * res = gf_res_reserve.get();

    const auto gparams = graph_params(res, ubatch, mctx, LLM_GRAPH_TYPE_DEFAULT);

    res->reset();

    auto * gf = model.build_graph(gparams);

    this->n_outputs = save_n_outputs;

    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {
        GGML_ASSERT(!sizes);
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    return gf;
}

std::string llama_dense_stream_meter::end(const ggml_backend_sched_kimi_copy_stats & now) {
    const uint64_t tensors = now.dense_tensors - start.dense_tensors;
    if (phase == nullptr || tensors == 0) {
        return {};
    }
    char line[320];
    snprintf(line, sizeof(line), "[dense_stream] phase=%s tokens=%" PRIu64 " tensors=%" PRIu64 " bytes=%" PRIu64
        " read_us=%" PRIu64 " first_us=%" PRIu64 " wait_us=%" PRIu64 " readahead=%" PRIu64 " fallbacks=%" PRIu64,
        phase, tokens, tensors, now.dense_bytes - start.dense_bytes, now.dense_read_us - start.dense_read_us,
        now.dense_first_us - start.dense_first_us, now.dense_wait_us - start.dense_wait_us,
        now.dense_readahead - start.dense_readahead, now.dense_fallbacks - start.dense_fallbacks);
    return line;
}

std::string llama_prefill_copy_meter::end(const ggml_backend_sched_kimi_copy_stats & now) {
    if (passes == 0) {
        return {};
    }
    const uint64_t bytes = now.bytes - start.bytes;
    const uint64_t read_us = now.read_us - start.read_us;
    char line[256];
    snprintf(line, sizeof(line), "[moe_prefill_copy] tokens=%" PRIu64 " passes=%u expert_slices=%" PRIu64 " bytes=%"
        PRIu64 " read_us=%" PRIu64 " read_gbs=%.2f fallbacks=%" PRIu64, tokens, passes,
        now.expert_slices - start.expert_slices, bytes, read_us, read_us ? bytes / 1e3 / read_us : 0.0,
        now.fallbacks - start.fallbacks);
    passes = 0;
    tokens = 0;
    return line;
}

bool llama_gpu_offloads_mul_mat_id(const std::vector<ggml_backend_dev_t> & gpus,
        const std::vector<const ggml_tensor *> & experts, int64_t n_expert_used, uint32_t n_tokens) {
    if (gpus.empty() || experts.empty() || n_tokens == 0) {
        return false;
    }
    // the op over each expert tensor, without data: the tests read only types, shapes and buffers
    const ggml_init_params params = { (1 + 2 * experts.size()) * ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr ctx { ggml_init(params) };
    ggml_tensor * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_expert_used, n_tokens);
    std::vector<const ggml_tensor *> ops;
    for (const ggml_tensor * as : experts) {
        ggml_tensor * b = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, as->ne[0], 1, n_tokens);
        ops.push_back(ggml_mul_mat_id(ctx.get(), const_cast<ggml_tensor *>(as), b, ids));
    }
    for (ggml_backend_dev_t dev : gpus) {
        if (std::all_of(ops.begin(), ops.end(), [&](const ggml_tensor * op) {
                return ggml_backend_dev_supports_op(dev, op) && ggml_backend_dev_offload_op(dev, op); })) {
            return true;
        }
    }
    return false;
}

bool llama_context::gpu_offloads_experts(uint32_t n_tokens) const {
    if (!cparams.op_offload) {
        return false;
    }
    std::vector<ggml_backend_dev_t> gpus;
    for (const auto & backend : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
        const auto type = dev ? ggml_backend_dev_type(dev) : GGML_BACKEND_DEVICE_TYPE_CPU;
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            gpus.push_back(dev);
        }
    }
    // one expert tensor of each type and buffer type: the test depends on them, not on the layer
    auto buft = [](const ggml_tensor * t) { return t->buffer ? ggml_backend_buffer_get_type(t->buffer) : nullptr; };
    std::vector<const ggml_tensor *> experts;
    for (const auto & layer : model.layers) {
        for (const ggml_tensor * t : { layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps, layer.ffn_gate_up_exps }) {
            if (t && std::none_of(experts.begin(), experts.end(), [&](const ggml_tensor * e) {
                    return e->type == t->type && buft(e) == buft(t); })) {
                experts.push_back(t);
            }
        }
    }
    return llama_gpu_offloads_mul_mat_id(gpus, experts, model.hparams.n_expert_used, n_tokens);
}

bool llama_context::cpu_sink_prefill_route(uint32_t n_tokens) const {
    return cpu_sink_prefill_route(n_tokens, cpu_prefill_request_max_tokens);
}

bool llama_context::cpu_sink_prefill_route(uint32_t n_tokens, uint32_t max_tokens) const {
    return llama_cpu_sink_prefill_route(cpu_sink_enabled, max_tokens, n_tokens,
        cpu_sink_enabled && n_tokens > max_tokens && gpu_offloads_experts(n_tokens));
}

uint32_t llama_prefill_ubatch_search(uint32_t target, uint32_t granularity, uint32_t & max_sink_tokens,
        const std::function<bool(uint32_t, uint32_t)> & sink_route, const std::function<bool(uint32_t)> & reserve) {
    if (reserve(target)) {
        return target;
    }
    const uint32_t step = std::min<uint32_t>(std::max<uint32_t>(256, granularity), std::max<uint32_t>(1, target / 2));
    // the largest size in (lo, target) that reserve accepts, and the smallest it refused; with graph, a size that
    // takes the sink is below every graph size (the route is monotone in the size)
    auto largest = [&](uint32_t lo, bool graph) {
        uint32_t found = 0, failed = target;
        while (failed > lo + step) {
            uint32_t probe = lo + (failed - lo) / 2;
            probe = std::max(lo + step, probe - probe % step);
            if (probe >= failed) { break; }
            if (graph && sink_route(probe, max_sink_tokens)) {
                lo = probe;
            } else if (reserve(probe)) {
                lo = found = probe;
            } else {
                failed = probe;
            }
        }
        return std::make_pair(found, failed);
    };
    if (!sink_route(target, max_sink_tokens)) {
        if (const uint32_t graph = largest(std::min(max_sink_tokens, target), true).first) {
            return graph;
        }
        max_sink_tokens = target;
        if (reserve(target)) {
            return target;
        }
    }
    const auto [sink, failed] = largest(0, false);
    if (!sink && failed > 1 && reserve(1)) {
        return 1;
    }
    return sink;
}

llm_graph_params llama_context::graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const {
    return {
        /*.arch        =*/ model.arch,
        /*.hparams     =*/ model.hparams,
        /*.cparams     =*/ cparams,
        /*.ubatch      =*/ ubatch,
        /*.gtype       =*/ gtype,
        /*.sched       =*/ sched.get(),
        /*.backend_cpu =*/ backend_cpu,
        /*.cvec        =*/ cvec.get(),
        /*.loras       =*/ loras.get(),
        /*.mctx        =*/ mctx,
        /*.cross       =*/ &cross,
        /*.samplers    =*/ sampling.samplers,
        /*.n_outputs   =*/ n_outputs,
        /*.cpu_sink_decode =*/ cpu_sink_enabled && cpu_sink_decode_active && !cparams.warmup,
        /*.cpu_sink_prefill =*/ !cpu_sink_decode_active && !cparams.warmup && cpu_sink_prefill_route(ubatch.n_tokens),
        /*.cpu_sink_guard =*/ k3_guard_active,
        /*.cpu_sink_runtime =*/ cpu_sink_runtime.get(),
        /*.k3_expert_policy =*/ k3_runtime ? (k3_guard_active ? k3_runtime->guard : k3_runtime->primary) : llama_k3_decode_expert_policy{},
        /*.k3_expert_table =*/ k3_runtime && !k3_guard_active ? k3_runtime->table : llama_k3_decode_expert_table{},
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,
        /*.n_ctx_reserve =*/ sched_reserve_n_ctx ? sched_reserve_n_ctx : cparams.n_ctx,
    };
}

ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    // The sink owns its persistent compute/I/O pool. Keep the other ggml CPU
    // workers asleep during its custom op instead of spinning at its barrier.
    const bool cpu_sink_graph = gf_res_prev &&
        gf_res_prev->get_gf() == gf && !gf_res_prev->cpu_sink_nodes.empty();
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        const bool sink_cpu_backend = cpu_sink_graph &&
            set_n_threads_fn.first == backend_cpu;
        set_n_threads_fn.second(set_n_threads_fn.first, sink_cpu_backend ? 1 : n_threads);
    }

    const bool graph_profile = kimi_graph_profile_enabled();
    const int64_t t_submit_start_us = graph_profile ? ggml_time_us() : 0;
    using activation_quant_reuse_fn = void (*)(ggml_backend_t);
    using activation_quant_reuse_scope = std::pair<ggml_backend_t, activation_quant_reuse_fn>;
    std::vector<activation_quant_reuse_scope> activation_quant_reuse_scopes;
    static const bool activation_quant_reuse_enabled = []() {
        const char * value = std::getenv("GGML_CUDA_DENSE_ACTIVATION_QUANT_REUSE");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0 &&
            std::strcmp(value, "false") != 0;
    }();
    if (activation_quant_reuse_enabled) {
        for (const auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg == nullptr) {
                continue;
            }
            auto begin = reinterpret_cast<activation_quant_reuse_fn>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_activation_quant_reuse_begin"));
            auto end = reinterpret_cast<activation_quant_reuse_fn>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_activation_quant_reuse_end"));
            if (begin != nullptr && end != nullptr) {
                begin(backend.get());
                activation_quant_reuse_scopes.emplace_back(backend.get(), end);
            }
        }
    }
    ggml_backend_sched_kimi_set_profile_phase(batched ? 1 : 2);
    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    ggml_backend_sched_kimi_set_profile_phase(0);
    for (auto scope = activation_quant_reuse_scopes.rbegin();
            scope != activation_quant_reuse_scopes.rend(); ++scope) {
        scope->second(scope->first);
    }
    if (graph_profile) {
        const uint64_t submit_us = (uint64_t) (ggml_time_us() - t_submit_start_us);
        g_kimi_graph_profile.submit_calls.fetch_add(1, std::memory_order_relaxed);
        g_kimi_graph_profile.submit_us.fetch_add(submit_us, std::memory_order_relaxed);
    }
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    } else if (ds4_weight_profile_path() != nullptr) {
        ggml_backend_sched_synchronize(sched.get());
        FILE * fp = ds4_weight_profile_fp();
        static uint64_t graph_seq = 0;
        const uint64_t seq = graph_seq++;
        if (fp != nullptr) {
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                ggml_tensor * node = ggml_graph_node(gf, i);
                if (node == nullptr || std::strstr(node->name, "ffn_weights") == nullptr) {
                    continue;
                }
                if (node->type != GGML_TYPE_F32 || ggml_nelements(node) <= 0 || ggml_nelements(node) > 4096) {
                    continue;
                }
                ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), node);
                if (backend == nullptr) {
                    continue;
                }
                const int il = ds4_weight_profile_layer_from_name(node->name);
                const int64_t ne0 = node->ne[0] > 0 ? node->ne[0] : 1;
                const int64_t ne1 = node->ne[1] > 0 ? node->ne[1] : 1;
                const int64_t ne2 = node->ne[2] > 0 ? node->ne[2] : 1;
                for (int64_t token = 0; token < ne2; ++token) {
                    for (int64_t rank = 0; rank < ne1; ++rank) {
                        float value = 0.0f;
                        const size_t offs = (size_t) (token * node->nb[2] + rank * node->nb[1]);
                        ggml_backend_tensor_get(node, &value, offs, sizeof(float));
                        std::fprintf(fp, "%llu,%d,%s,%d,%lld,%lld,%lld,%zu,%zu,%zu,%lld,%lld,%.9g\n",
                                (unsigned long long) seq,
                                batched ? 1 : 0,
                                node->name,
                                il,
                                (long long) ne0,
                                (long long) ne1,
                                (long long) ne2,
                                (size_t) node->nb[0],
                                (size_t) node->nb[1],
                                (size_t) node->nb[2],
                                (long long) token,
                                (long long) rank,
                                value);
                    }
                }
            }
            std::fflush(fp);
        }
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        // At CUDA's automatic offload threshold (currently 32 rows), a
        // host-backed MUL_MAT_ID would otherwise be moved wholesale to CUDA.
        // That bypasses the grouped prefill provider and faults the multi-GiB
        // GGUF expert tensors into the cgroup.  Keep only the three routed MoE
        // operations on the CPU scheduler while the request-scoped workspace
        // is active; the Kimi CPU entry point immediately dispatches their
        // active rows to the grouped CUDA path.  sched_reserve() runs before
        // phase_workspace_active is set, so it still reserves the observed
        // target CUDA workspace that the cache transaction budgeted for.
        const bool phase_grouped_moe =
            phase_workspace_active && ubatch.n_tokens > 1 &&
            cur->op == GGML_OP_MUL_MAT_ID &&
            (std::strcmp(name, "ffn_moe_gate") == 0 ||
             std::strcmp(name, "ffn_moe_up") == 0 ||
             std::strcmp(name, "ffn_moe_down") == 0);
        if (phase_grouped_moe) {
            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend_cpu);
        }
        if (ds4_weight_profile_path() != nullptr && std::strcmp(name, "ffn_weights") == 0) {
            ggml_set_output(cur);
        }

        // norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && strcmp(name, "norm") == 0) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy() = default;

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(const ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    size_t size_written = 0;
};

class llama_io_write_buffer : public llama_io_write_i {
public:
    llama_io_write_buffer(
            uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(const ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        ggml_backend_tensor_get(tensor, ptr, offset, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;
};

class llama_io_read_buffer : public llama_io_read_i {
public:
    llama_io_read_buffer(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    const uint8_t * read(size_t size) override {
        const uint8_t * base_ptr = ptr;
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        ptr += size;
        size_read += size;
        buf_size -= size;
        return base_ptr;
    }

    void read_to(void * dst, size_t size) override {
        memcpy(dst, read(size), size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(const ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read_to(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    const uint8_t * read(size_t size) override {
        temp_buffer.resize(size);
        read_to(temp_buffer.data(), size);
        return temp_buffer.data();
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

// streams to and from a FILE * the caller positioned, e.g. inside a container file of its own
class llama_io_write_stdio : public llama_io_write_i {
public:
    llama_io_write_stdio(FILE * fp) : fp(fp) {}

    void write(const void * src, size_t size) override {
        if (size > 0 && std::fwrite(src, 1, size, fp) != size) {
            throw std::runtime_error("failed to write state");
        }
        size_written += size;
    }

    void write_tensor(const ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    FILE * fp;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_stdio : public llama_io_read_i {
public:
    llama_io_read_stdio(FILE * fp) : fp(fp) {}

    void read_to(void * dst, size_t size) override {
        if (size > 0 && std::fread(dst, 1, size, fp) != size) {
            throw std::runtime_error("unexpectedly reached end of state");
        }
        size_read += size;
    }

    const uint8_t * read(size_t size) override {
        temp_buffer.resize(size);
        read_to(temp_buffer.data(), size);
        return temp_buffer.data();
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    FILE * fp;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

size_t llama_context::state_get_size() {
    llama_io_write_dummy io;
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_buffer io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_buffer io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    llama_io_write_dummy io;
    try {
        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    llama_io_write_buffer io(dst, size);
    try {
        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    llama_io_read_buffer io(src, size);
    try {
        return state_seq_read_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

uint32_t llama_context::state_seq_at_align() {
    return memory ? memory->state_write_at_align() : 0;
}

bool llama_context::state_seq_can_save_at(llama_seq_id seq_id, int32_t n_tokens) {
    return n_tokens < 0 || (memory && memory->state_can_write_at(seq_id, n_tokens));
}

llama_pos llama_context::state_seq_pos_min_at(llama_seq_id seq_id, int32_t n_tokens) {
    return memory ? memory->state_pos_min_at(seq_id, n_tokens) : n_tokens - 1;
}

size_t llama_context::state_seq_get_size_at(llama_seq_id seq_id, llama_state_seq_flags flags, int32_t n_tokens) {
    llama_io_write_dummy io;
    try {
        return state_seq_write_data(io, seq_id, flags, n_tokens);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data_at(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags, int32_t n_tokens) {
    llama_io_write_buffer io(dst, size);
    try {
        return state_seq_write_data(io, seq_id, flags, n_tokens);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_fwrite_at(llama_seq_id seq_id, FILE * fp, llama_state_seq_flags flags, int32_t n_tokens) {
    llama_io_write_stdio io(fp);
    try {
        return state_seq_write_data(io, seq_id, flags, n_tokens);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error writing state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_fread(llama_seq_id seq_id, FILE * fp, llama_state_seq_flags flags) {
    llama_io_read_stdio io(fp);
    try {
        return state_seq_read_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error reading state: %s\n", __func__, err.what());
        return 0;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id, 0);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, 0);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        memory->state_read(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags, int32_t n_tokens) {
    GGML_UNUSED(seq_id);

    if (memory) {
        if (n_tokens < 0) {
            memory->state_write(io, seq_id, flags);
        } else {
            memory->state_write_at(io, seq_id, flags, n_tokens);
        }
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(seq_id);

    if (memory) {
        memory->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;
}

llama_memory_breakdown llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (model.hparams.no_alloc) {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += backend_buf_exp_size[i];
        }
    } else {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr.get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
    }
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model->output,          param_filter, param_filter_ud);
    llama_set_param(model->output_b,        param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls,             param_filter, param_filter_ud);
    llama_set_param(model->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b,       param_filter, param_filter_ud);
    llama_set_param(model->cls_norm,        param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        if (!balloc->init(batch, model.vocab, nullptr, model.hparams.n_embd_inp(), cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
            LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
            return;
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            auto * res = gf_res_prev.get();

            const auto gparams = graph_params(res, ubatch, mctx.get(), LLM_GRAPH_TYPE_DEFAULT);

            res->reset();

            auto * gf = model.build_graph(gparams);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
        /*.kv_unified                  =*/ false,
        /*.sampler                     =*/ nullptr,
        /*.n_sampler                   =*/ 0,
        /*.ctx_other                   =*/ nullptr,
    };

    return result;
}

llama_context * llama_init_from_model(
                 llama_model * model,
        llama_context_params   params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (model->split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            LLAMA_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
        if (ggml_is_quantized(params.type_k) || ggml_is_quantized(params.type_v)) {
            LLAMA_LOG_ERROR("%s: simultaneous use of SPLIT_MODE_TENSOR and KV cache quantization not implemented\n", __func__);
            return nullptr;
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = ggml_blck_size(params.type_k);
        for (uint32_t il = 0; il < model->hparams.n_layer; ++il) {
            if (model->hparams.n_embd_head_k(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                    __func__, ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k(il));
                return nullptr;
            }
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = ggml_blck_size(params.type_v);
        for (uint32_t il = 0; il < model->hparams.n_layer; ++il) {
            if (model->hparams.n_embd_head_v(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n",
                    __func__, ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v(il));
                return nullptr;
            }
        }
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
        LLAMA_LOG_ERROR("%s: V cache quantization requires flash_attn\n", __func__);
        return nullptr;
    }

    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&
        params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    try {
        auto * ctx = new llama_context(*model, params);
        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

// deprecated
llama_context * llama_new_context_with_model(
                 llama_model * model,
        llama_context_params   params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

extern "C" bool llama_moe_prefill_phase_workspace(
        llama_context * ctx,
        int active) {
    return ctx && ctx->moe_prefill_phase_workspace(active != 0);
}

extern "C" bool llama_moe_prefill_phase_workspace_sized(
        llama_context * ctx,
        int active,
        uint32_t requested_tokens) {
    return ctx && ctx->moe_prefill_phase_workspace(active != 0, requested_tokens);
}

extern "C" bool llama_moe_prefill_phase_workspace_sized_ex(
    llama_context * ctx, int active, uint32_t prompt_tokens, uint32_t prefill_tokens) {
    return ctx && ctx->moe_prefill_phase_workspace(active != 0, prompt_tokens, prefill_tokens);
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_embeddings_nextn(llama_context * ctx, bool value, bool masked) {
    ctx->set_embeddings_nextn(value, masked);
}

void llama_set_embeddings_layer_inp(llama_context * ctx, uint32_t lid, bool value) {
    ctx->set_embeddings_layer_inp(lid, value);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

float * llama_get_embeddings_nextn(llama_context * ctx) {
    ctx->synchronize();
    return ctx->get_embeddings_nextn();
}

float * llama_get_embeddings_nextn_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();
    return ctx->get_embeddings_nextn_ith(i);
}

float * llama_get_embeddings_layer_inp(llama_context * ctx, uint32_t lid) {
    ctx->synchronize();
    return ctx->get_embeddings_layer_inp(lid);
}

llama_context * llama_get_ctx_other(llama_context * ctx) {
    return ctx->get_cparams().ctx_other;
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_token_ith(i);
}

float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_probs_ith(i);
}

float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_logits_ith(i);
}

llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return const_cast<llama_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t llama_get_sampled_candidates_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs) {
    auto * memory = ctx->get_memory();
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
    }
    return ctx->graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
}

// llama adapter API

int32_t llama_set_adapters_lora(
            llama_context * ctx,
            llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales) {
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_adapters_lora call");
    }

    ctx->set_adapters_lora(adapters, n_adapters, scales);

    return 0;
}

int32_t llama_set_adapter_cvec(
        llama_context * ctx,
          const float * data,
               size_t   len,
              int32_t   n_embd,
              int32_t   il_start,
              int32_t   il_end) {
    bool res = ctx->set_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    return ctx->get_memory();
}

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
}

bool llama_memory_seq_rm(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return true;
    }

    return mem->seq_rm(seq_id, p0, p1);
}

void llama_memory_seq_cp(
        llama_memory_t mem,
          llama_seq_id seq_id_src,
          llama_seq_id seq_id_dst,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_seq_keep(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void llama_memory_seq_add(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
             llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void llama_memory_seq_div(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
                   int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_seq_pos_min(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}

size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

uint32_t llama_state_seq_at_align(llama_context * ctx) {
    return ctx->state_seq_at_align();
}

bool llama_state_seq_can_save_at(llama_context * ctx, llama_seq_id seq_id, int32_t n_tokens) {
    return ctx->state_seq_can_save_at(seq_id, n_tokens);
}

llama_pos llama_state_seq_pos_min_at(llama_context * ctx, llama_seq_id seq_id, int32_t n_tokens) {
    return ctx->state_seq_pos_min_at(seq_id, n_tokens);
}

bool llama_host_reserve(llama_context * ctx, size_t bytes) {
    return ctx->host_reserve(bytes);
}

size_t llama_state_seq_get_size_at(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags, int32_t n_tokens) {
    return ctx->state_seq_get_size_at(seq_id, flags, n_tokens);
}

size_t llama_state_seq_get_data_at(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags, int32_t n_tokens) {
    ctx->synchronize();

    return ctx->state_seq_get_data_at(seq_id, dst, size, flags, n_tokens);
}

size_t llama_state_seq_fwrite_at(llama_context * ctx, FILE * fp, llama_seq_id seq_id, llama_state_seq_flags flags, int32_t n_tokens) {
    ctx->synchronize();

    return ctx->state_seq_fwrite_at(seq_id, fp, flags, n_tokens);
}

size_t llama_state_seq_fread_ext(llama_context * ctx, FILE * fp, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_fread(seq_id, fp, flags);
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

///

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}

//
// ext
//

llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx) {
    return ctx->memory_breakdown();
}
