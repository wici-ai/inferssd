#include "common.cuh"
#include "convert.cuh"
#include "ggml-quants.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

extern "C" bool ggml_cuda_kimi_dense_stream_mul_mat(
    int src0_type_int,
    const char * src0_name,
    const void * src0_data,
    size_t src0_bytes,
    int64_t ne00,
    int64_t ne01,
    int64_t ne02,
    int64_t ne03,
    size_t nb01,
    size_t nb02,
    size_t nb03,
    const float * src1_data,
    int64_t ne11,
    int64_t ne12,
    int64_t ne13,
    size_t nb11,
    size_t nb12,
    size_t nb13,
    float * dst_data,
    size_t nb1,
    size_t nb2,
    size_t nb3);

extern "C" bool ggml_cuda_kimi_dense_stream_should_intercept(const char * tensor_name);

#ifdef GGML_CUDA_MOE_STREAM

extern "C" bool ggml_cuda_moe_stream_mmvq_dev(
    int src0_type_int,
    const void * d_src0,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    const float * d_src1_f32,
    void * d_src1_q8,
    float * d_dst,
    cudaStream_t stream);

namespace {

struct kimi_dense_stream_profile {
    uint64_t calls = 0;
    uint64_t chunks = 0;
    uint64_t prefetch_hits = 0;
    uint64_t weight_bytes = 0;
    uint64_t direct_weight_bytes = 0;
    uint64_t direct_weight_calls = 0;
    double host_stage_ms = 0.0;
    double weight_h2d_ms = 0.0;
    double weight_exposed_ms = 0.0;
    double input_h2d_ms = 0.0;
    double compute_ms = 0.0;
    double launch_ms = 0.0;
    double output_d2h_ms = 0.0;
    double scatter_ms = 0.0;
    double total_ms = 0.0;
};

struct kimi_dense_stream_cgroup_snapshot {
    bool valid = false;
    uint64_t current = 0;
    uint64_t events_max = 0;
    uint64_t events_oom = 0;
    uint64_t events_oom_kill = 0;
    uint64_t events_oom_group_kill = 0;
};

struct kimi_dense_stream_registered_range {
    uintptr_t begin = 0;
    uintptr_t end = 0;
};

struct kimi_dense_stream_registration {
    bool direct = false;
    bool already_pinned = false;
    bool retried_without_read_only = false;
    size_t source_bytes = 0;
    size_t registered_bytes = 0;
    double register_ms = 0.0;
    cudaError_t error = cudaSuccess;
    kimi_dense_stream_cgroup_snapshot before;
    kimi_dense_stream_cgroup_snapshot after;
};

struct kimi_dense_stream_tensor_desc {
    int type = GGML_TYPE_COUNT;
    const void * data = nullptr;
    size_t bytes = 0;
    int64_t ne00 = 0;
    int64_t ne01 = 0;
    size_t nb01 = 0;
};

struct kimi_dense_stream_pipeline_buffer {
    void * device = nullptr;
    size_t capacity = 0;
    cudaEvent_t copy_start = nullptr;
    cudaEvent_t copy_done = nullptr;
    cudaEvent_t compute_done = nullptr;
    bool has_copy = false;
    bool has_compute = false;
    std::string tensor;
    size_t row_begin = 0;
    size_t rows = 0;
    size_t bytes = 0;
};

struct kimi_dense_stream_compute_events {
    cudaEvent_t start = nullptr;
    cudaEvent_t done = nullptr;
};

struct kimi_dense_stream_state {
    std::mutex mutex;
    bool initialized = false;
    bool available = false;
    bool report_registered = false;
    bool zero_copy = false;
    bool zero_copy_register_only = false;
    bool pipeline = false;
    bool debug = false;
    std::atomic<bool> pipeline_started { false };
    std::string zero_copy_tensor;
    size_t pipeline_chunk_bytes = 32ULL * 1024 * 1024;
    int64_t pipeline_max_batch = 16;

    cudaStream_t stream = nullptr;
    cudaStream_t transfer_stream = nullptr;
    cublasHandle_t cublas = nullptr;
    cudaEvent_t event_weight_start = nullptr;
    cudaEvent_t event_weight_done = nullptr;
    cudaEvent_t event_input_done = nullptr;
    cudaEvent_t event_compute_done = nullptr;
    cudaEvent_t event_output_done = nullptr;
    std::vector<kimi_dense_stream_compute_events> pipeline_compute_events;
    kimi_dense_stream_pipeline_buffer pipeline_buffers[2];

    void * host_weight = nullptr;
    size_t host_weight_size = 0;
    float * host_input = nullptr;
    size_t host_input_size = 0;
    void * host_input_quantized = nullptr;
    size_t host_input_quantized_size = 0;
    float * host_output = nullptr;
    size_t host_output_size = 0;

    void * device_weight = nullptr;
    size_t device_weight_size = 0;
    float * device_weight_f32 = nullptr;
    size_t device_weight_f32_size = 0;
    float * device_input = nullptr;
    size_t device_input_size = 0;
    __half * device_input_f16 = nullptr;
    size_t device_input_f16_size = 0;
    void * device_input_q8 = nullptr;
    size_t device_input_q8_size = 0;
    float * device_output = nullptr;
    size_t device_output_size = 0;

    size_t peak_device_bytes = 0;
    size_t peak_pinned_bytes = 0;
    size_t registered_bytes = 0;
    double register_ms = 0.0;
    bool have_registration_cgroup_before = false;
    kimi_dense_stream_cgroup_snapshot registration_cgroup_before;
    kimi_dense_stream_cgroup_snapshot registration_cgroup_after;
    std::string cgroup_dir;
    std::vector<kimi_dense_stream_registered_range> registered_ranges;
    std::map<const void *, kimi_dense_stream_registration> registrations;
    std::vector<std::string> pipeline_order;
    std::map<std::string, kimi_dense_stream_tensor_desc> pipeline_tensors;
    bool pipeline_order_ready = false;
    std::map<std::string, kimi_dense_stream_profile> profiles;
};

static kimi_dense_stream_state & kimi_dense_stream_get_state() {
    static kimi_dense_stream_state * state = new kimi_dense_stream_state();
    return *state;
}

static bool kimi_dense_stream_env_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] && std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0;
}

#if defined(__linux__)
static bool kimi_dense_stream_read_u64(const std::string & path, uint64_t * value) {
    FILE * file = std::fopen(path.c_str(), "r");
    if (!file) {
        return false;
    }

    unsigned long long parsed = 0;
    const bool ok = std::fscanf(file, "%llu", &parsed) == 1;
    std::fclose(file);
    if (ok) {
        *value = (uint64_t) parsed;
    }
    return ok;
}

static std::string kimi_dense_stream_find_cgroup_dir() {
    FILE * file = std::fopen("/proc/self/cgroup", "r");
    if (!file) {
        return {};
    }

    char line[4096];
    std::string result;
    while (std::fgets(line, sizeof(line), file)) {
        char * path = std::strstr(line, "0::");
        if (!path) {
            continue;
        }
        path += 3;
        path[std::strcspn(path, "\r\n")] = '\0';
        result = "/sys/fs/cgroup";
        result += path;
        break;
    }
    std::fclose(file);
    return result;
}

static kimi_dense_stream_cgroup_snapshot kimi_dense_stream_read_cgroup_snapshot(
        kimi_dense_stream_state & state) {
    kimi_dense_stream_cgroup_snapshot snapshot;
    if (state.cgroup_dir.empty()) {
        state.cgroup_dir = kimi_dense_stream_find_cgroup_dir();
    }
    if (state.cgroup_dir.empty() ||
            !kimi_dense_stream_read_u64(state.cgroup_dir + "/memory.current", &snapshot.current)) {
        return snapshot;
    }

    FILE * file = std::fopen((state.cgroup_dir + "/memory.events").c_str(), "r");
    if (!file) {
        return snapshot;
    }

    char name[64];
    unsigned long long value = 0;
    while (std::fscanf(file, "%63s %llu", name, &value) == 2) {
        if (std::strcmp(name, "max") == 0) {
            snapshot.events_max = (uint64_t) value;
        } else if (std::strcmp(name, "oom") == 0) {
            snapshot.events_oom = (uint64_t) value;
        } else if (std::strcmp(name, "oom_kill") == 0) {
            snapshot.events_oom_kill = (uint64_t) value;
        } else if (std::strcmp(name, "oom_group_kill") == 0) {
            snapshot.events_oom_group_kill = (uint64_t) value;
        }
    }
    std::fclose(file);
    snapshot.valid = true;
    return snapshot;
}
#else
static kimi_dense_stream_cgroup_snapshot kimi_dense_stream_read_cgroup_snapshot(kimi_dense_stream_state &) {
    return {};
}
#endif

static void kimi_dense_stream_add_registered_range(
        kimi_dense_stream_state & state, uintptr_t begin, uintptr_t end) {
    kimi_dense_stream_registered_range merged = { begin, end };
    std::vector<kimi_dense_stream_registered_range> ranges;
    ranges.reserve(state.registered_ranges.size() + 1);
    bool inserted = false;
    for (const kimi_dense_stream_registered_range & range : state.registered_ranges) {
        if (range.end <= merged.begin) {
            ranges.push_back(range);
        } else if (merged.end <= range.begin) {
            if (!inserted) {
                ranges.push_back(merged);
                inserted = true;
            }
            ranges.push_back(range);
        } else {
            merged.begin = std::min(merged.begin, range.begin);
            merged.end = std::max(merged.end, range.end);
        }
    }
    if (!inserted) {
        ranges.push_back(merged);
    }
    state.registered_ranges.swap(ranges);
}

static bool kimi_dense_stream_register_weight(
        kimi_dense_stream_state & state,
        const char * name,
        const void * data,
        size_t size) {
    const auto found = state.registrations.find(data);
    if (found != state.registrations.end()) {
        return found->second.direct;
    }

    kimi_dense_stream_registration registration;
    registration.source_bytes = size;
    registration.before = kimi_dense_stream_read_cgroup_snapshot(state);
    if (!state.have_registration_cgroup_before && registration.before.valid) {
        state.registration_cgroup_before = registration.before;
        state.have_registration_cgroup_before = true;
    }

#if defined(__linux__)
    const long page_size_long = sysconf(_SC_PAGESIZE);
    const uintptr_t page_size = page_size_long > 0 ? (uintptr_t) page_size_long : 4096;
    const uintptr_t source_begin = (uintptr_t) data;
    const uintptr_t source_end = source_begin + size;
    const uintptr_t begin = source_begin & ~(page_size - 1);
    const uintptr_t end = (source_end + page_size - 1) & ~(page_size - 1);

    std::vector<kimi_dense_stream_registered_range> gaps;
    uintptr_t cursor = begin;
    for (const kimi_dense_stream_registered_range & range : state.registered_ranges) {
        if (range.end <= cursor) {
            continue;
        }
        if (range.begin >= end) {
            break;
        }
        if (range.begin > cursor) {
            gaps.push_back({ cursor, std::min(range.begin, end) });
        }
        cursor = std::max(cursor, range.end);
        if (cursor >= end) {
            break;
        }
    }
    if (cursor < end) {
        gaps.push_back({ cursor, end });
    }

    unsigned int flags = cudaHostRegisterDefault;
#if CUDART_VERSION >= 11020
    flags |= cudaHostRegisterReadOnly;
#endif
    const auto register_start = std::chrono::steady_clock::now();
    for (const kimi_dense_stream_registered_range & gap : gaps) {
        cudaError_t error = cudaHostRegister((void *) gap.begin, gap.end - gap.begin, flags);
#if CUDART_VERSION >= 11020
        if (error == cudaErrorNotSupported && (flags & cudaHostRegisterReadOnly)) {
            cudaGetLastError();
            registration.retried_without_read_only = true;
            error = cudaHostRegister(
                (void *) gap.begin, gap.end - gap.begin, cudaHostRegisterDefault);
        }
#endif
        if (error == cudaErrorHostMemoryAlreadyRegistered) {
            cudaGetLastError();
            error = cudaSuccess;
        }
        if (error != cudaSuccess) {
            registration.error = error;
            break;
        }
        registration.registered_bytes += gap.end - gap.begin;
        kimi_dense_stream_add_registered_range(state, gap.begin, gap.end);
    }
    const auto register_done = std::chrono::steady_clock::now();
    registration.register_ms =
        std::chrono::duration<double, std::milli>(register_done - register_start).count();
    if (registration.error != cudaSuccess && registration.registered_bytes == 0) {
        cudaPointerAttributes attributes = {};
        cudaGetLastError();
        if (cudaPointerGetAttributes(&attributes, data) == cudaSuccess &&
                attributes.type == cudaMemoryTypeHost) {
            registration.error = cudaSuccess;
            registration.already_pinned = true;
        } else {
            cudaGetLastError();
        }
    }
    registration.direct = registration.error == cudaSuccess;
#else
    registration.error = cudaErrorNotSupported;
#endif

    registration.after = kimi_dense_stream_read_cgroup_snapshot(state);
    if (registration.after.valid) {
        state.registration_cgroup_after = registration.after;
    }
    state.registered_bytes += registration.registered_bytes;
    state.register_ms += registration.register_ms;
    state.registrations[data] = registration;

    const int64_t current_delta = registration.before.valid && registration.after.valid ?
        (int64_t) registration.after.current - (int64_t) registration.before.current : 0;
    std::fprintf(
        stderr,
        "[kimi_dense_stream] register: tensor=%s direct=%d source_bytes=%llu registered_bytes=%llu "
        "register_ms=%.3f already_pinned=%d retried_without_read_only=%d cuda_error=%d "
        "memory_current_before=%llu memory_current_after=%llu "
        "memory_current_delta=%lld events_max_before=%llu events_max_after=%llu "
        "events_oom_before=%llu events_oom_after=%llu events_oom_kill_before=%llu events_oom_kill_after=%llu\n",
        name,
        registration.direct ? 1 : 0,
        (unsigned long long) registration.source_bytes,
        (unsigned long long) registration.registered_bytes,
        registration.register_ms,
        registration.already_pinned ? 1 : 0,
        registration.retried_without_read_only ? 1 : 0,
        (int) registration.error,
        (unsigned long long) registration.before.current,
        (unsigned long long) registration.after.current,
        (long long) current_delta,
        (unsigned long long) registration.before.events_max,
        (unsigned long long) registration.after.events_max,
        (unsigned long long) registration.before.events_oom,
        (unsigned long long) registration.after.events_oom,
        (unsigned long long) registration.before.events_oom_kill,
        (unsigned long long) registration.after.events_oom_kill);
    if (!registration.direct) {
        std::fprintf(
            stderr,
            "[kimi_dense_stream] cudaHostRegister failed for %s: %s; using pinned staging\n",
            name,
            cudaGetErrorString(registration.error));
        cudaGetLastError();
    }
    return registration.direct;
}

static void kimi_dense_stream_report() {
    kimi_dense_stream_state & state = kimi_dense_stream_get_state();
    std::lock_guard<std::mutex> lock(state.mutex);

    for (const auto & item : state.profiles) {
        const kimi_dense_stream_profile & profile = item.second;
        std::fprintf(
            stderr,
            "[kimi_dense_stream] profile: tensor=%s calls=%llu chunks=%llu prefetch_hits=%llu weight_bytes=%llu "
            "direct_weight_calls=%llu direct_weight_bytes=%llu "
            "host_stage_ms=%.3f weight_h2d_ms=%.3f weight_exposed_ms=%.3f input_h2d_ms=%.3f "
            "compute_ms=%.3f launch_ms=%.3f output_d2h_ms=%.3f scatter_ms=%.3f total_ms=%.3f\n",
            item.first.c_str(),
            (unsigned long long) profile.calls,
            (unsigned long long) profile.chunks,
            (unsigned long long) profile.prefetch_hits,
            (unsigned long long) profile.weight_bytes,
            (unsigned long long) profile.direct_weight_calls,
            (unsigned long long) profile.direct_weight_bytes,
            profile.host_stage_ms,
            profile.weight_h2d_ms,
            profile.weight_exposed_ms,
            profile.input_h2d_ms,
            profile.compute_ms,
            profile.launch_ms,
            profile.output_d2h_ms,
            profile.scatter_ms,
            profile.total_ms);
    }

    if (!state.profiles.empty()) {
        std::fprintf(
            stderr,
            "[kimi_dense_stream] memory: device_peak_mib=%.3f pinned_peak_mib=%.3f\n",
            state.peak_device_bytes / (1024.0 * 1024.0),
            state.peak_pinned_bytes / (1024.0 * 1024.0));
    }

    if (!state.registrations.empty()) {
        const int64_t current_delta = state.have_registration_cgroup_before &&
                state.registration_cgroup_after.valid ?
            (int64_t) state.registration_cgroup_after.current -
                (int64_t) state.registration_cgroup_before.current : 0;
        std::fprintf(
            stderr,
            "[kimi_dense_stream] registration_summary: tensors=%llu registered_bytes=%llu register_ms=%.3f "
            "memory_current_before=%llu memory_current_after=%llu memory_current_delta=%lld "
            "events_max_before=%llu events_max_after=%llu events_oom_before=%llu events_oom_after=%llu "
            "events_oom_kill_before=%llu events_oom_kill_after=%llu "
            "events_oom_group_kill_before=%llu events_oom_group_kill_after=%llu\n",
            (unsigned long long) state.registrations.size(),
            (unsigned long long) state.registered_bytes,
            state.register_ms,
            (unsigned long long) state.registration_cgroup_before.current,
            (unsigned long long) state.registration_cgroup_after.current,
            (long long) current_delta,
            (unsigned long long) state.registration_cgroup_before.events_max,
            (unsigned long long) state.registration_cgroup_after.events_max,
            (unsigned long long) state.registration_cgroup_before.events_oom,
            (unsigned long long) state.registration_cgroup_after.events_oom,
            (unsigned long long) state.registration_cgroup_before.events_oom_kill,
            (unsigned long long) state.registration_cgroup_after.events_oom_kill,
            (unsigned long long) state.registration_cgroup_before.events_oom_group_kill,
            (unsigned long long) state.registration_cgroup_after.events_oom_group_kill);
    }
}

static bool kimi_dense_stream_init(kimi_dense_stream_state & state) {
    if (state.initialized) {
        return state.available;
    }

    state.initialized = true;
    state.zero_copy = kimi_dense_stream_env_enabled("GGML_KIMI_DENSE_STREAM_ZERO_COPY");
    state.zero_copy_register_only =
        kimi_dense_stream_env_enabled("GGML_KIMI_DENSE_STREAM_ZERO_COPY_REGISTER_ONLY");
    state.pipeline = kimi_dense_stream_env_enabled("GGML_KIMI_DENSE_STREAM_PIPELINE");
    state.debug = kimi_dense_stream_env_enabled("GGML_KIMI_DENSE_STREAM_DEBUG");
    const char * chunk_mib = std::getenv("GGML_KIMI_DENSE_STREAM_CHUNK_MIB");
    if (chunk_mib && chunk_mib[0]) {
        char * end = nullptr;
        const unsigned long long value = std::strtoull(chunk_mib, &end, 10);
        if (end != chunk_mib && *end == '\0' && value >= 8 && value <= 32) {
            state.pipeline_chunk_bytes = (size_t) value * 1024 * 1024;
        }
    }
    const char * max_batch = std::getenv("GGML_KIMI_DENSE_STREAM_MAX_BATCH");
    if (max_batch && max_batch[0]) {
        char * end = nullptr;
        const long long value = std::strtoll(max_batch, &end, 10);
        if (end != max_batch && *end == '\0' && value >= 1 && value <= 512) {
            state.pipeline_max_batch = (int64_t) value;
        }
    }
    const char * zero_copy_tensor = std::getenv("GGML_KIMI_DENSE_STREAM_ZERO_COPY_TENSOR");
    if (zero_copy_tensor) {
        state.zero_copy_tensor = zero_copy_tensor;
    }

    if (cudaSetDevice(0) != cudaSuccess ||
            cudaStreamCreateWithFlags(&state.stream, cudaStreamNonBlocking) != cudaSuccess ||
            cublasCreate(&state.cublas) != CUBLAS_STATUS_SUCCESS ||
            cublasSetStream(state.cublas, state.stream) != CUBLAS_STATUS_SUCCESS ||
            cublasSetMathMode(state.cublas, CUBLAS_PEDANTIC_MATH) != CUBLAS_STATUS_SUCCESS ||
            cudaEventCreate(&state.event_weight_start) != cudaSuccess ||
            cudaEventCreate(&state.event_weight_done) != cudaSuccess ||
            cudaEventCreate(&state.event_input_done) != cudaSuccess ||
            cudaEventCreate(&state.event_compute_done) != cudaSuccess ||
            cudaEventCreate(&state.event_output_done) != cudaSuccess) {
        std::fprintf(stderr, "[kimi_dense_stream] CUDA initialization failed; using CPU fallback\n");
        cudaGetLastError();
        return false;
    }

    if (state.pipeline) {
        bool events_ok = state.zero_copy &&
            cudaStreamCreateWithFlags(&state.transfer_stream, cudaStreamNonBlocking) == cudaSuccess;
        for (size_t i = 0; events_ok && i < 2; ++i) {
            events_ok =
                cudaEventCreate(&state.pipeline_buffers[i].copy_start) == cudaSuccess &&
                cudaEventCreate(&state.pipeline_buffers[i].copy_done) == cudaSuccess &&
                cudaEventCreate(&state.pipeline_buffers[i].compute_done) == cudaSuccess;
        }
        if (!events_ok) {
            std::fprintf(
                stderr,
                "[kimi_dense_stream] pipeline initialization failed or zero-copy is disabled; using CPU fallback\n");
            cudaGetLastError();
            return false;
        }
    }
    if (!state.report_registered) {
        std::atexit(kimi_dense_stream_report);
        state.report_registered = true;
    }
    state.available = true;
    std::fprintf(
        stderr,
        "[kimi_dense_stream] enabled: zero_copy=%d register_only=%d pipeline=%d "
        "chunk_mib=%.3f max_batch=%lld zero_copy_tensor=%s\n",
        state.zero_copy ? 1 : 0,
        state.zero_copy_register_only ? 1 : 0,
        state.pipeline ? 1 : 0,
        state.pipeline_chunk_bytes / (1024.0 * 1024.0),
        (long long) state.pipeline_max_batch,
        state.zero_copy_tensor.empty() ? "all" : state.zero_copy_tensor.c_str());
    return true;
}

static bool kimi_dense_stream_ensure_device(void ** ptr, size_t * capacity, size_t size) {
    if (*capacity >= size) {
        return true;
    }
    if (*ptr != nullptr) {
        cudaFree(*ptr);
        *ptr = nullptr;
        *capacity = 0;
    }
    if (cudaMalloc(ptr, size) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    *capacity = size;
    return true;
}

static bool kimi_dense_stream_ensure_pinned(void ** ptr, size_t * capacity, size_t size) {
    if (*capacity >= size) {
        return true;
    }
    if (*ptr != nullptr) {
        cudaFreeHost(*ptr);
        *ptr = nullptr;
        *capacity = 0;
    }
    if (cudaHostAlloc(ptr, size, cudaHostAllocDefault) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    *capacity = size;
    return true;
}

static void kimi_dense_stream_update_peaks(kimi_dense_stream_state & state) {
    const size_t device_bytes =
        state.device_weight_size +
        state.device_weight_f32_size +
        state.pipeline_buffers[0].capacity +
        state.pipeline_buffers[1].capacity +
        state.device_input_size +
        state.device_input_f16_size +
        state.device_input_q8_size +
        state.device_output_size;
    const size_t pinned_bytes =
        state.host_weight_size + state.host_input_size + state.host_input_quantized_size + state.host_output_size;
    state.peak_device_bytes = std::max(state.peak_device_bytes, device_bytes);
    state.peak_pinned_bytes = std::max(state.peak_pinned_bytes, pinned_bytes);
}

static __global__ void kimi_dense_stream_f32_to_f16(const float * src, __half * dst, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2half(src[i]);
    }
}

static __device__ __forceinline__ void kimi_dense_stream_q4k_scale_min(
        const block_q4_K & block, int group, int * scale, int * minimum) {
    if (group < 4) {
        *scale = block.scales[group] & 63;
        *minimum = block.scales[group + 4] & 63;
    } else {
        *scale = (block.scales[group + 4] & 0x0f) |
            ((block.scales[group - 4] >> 6) << 4);
        *minimum = (block.scales[group + 4] >> 4) |
            ((block.scales[group] >> 6) << 4);
    }
}

static __global__ void kimi_dense_stream_q4k_q8k(
        const void * weights,
        size_t row_stride,
        const block_q8_K * input,
        int64_t ne00,
        size_t rows,
        size_t output_rows,
        float * output) {
    const size_t row = blockIdx.x;
    const size_t column = blockIdx.y;
    const int lane = threadIdx.x;
    if (row >= rows || lane >= 8) {
        return;
    }

    const block_q4_K * x =
        (const block_q4_K *) ((const char *) weights + row * row_stride);
    const int blocks = (int) (ne00 / QK_K);
    input += column * blocks;
    float sum = 0.0f;
    float minimum_sum = 0.0f;
    for (int ib = 0; ib < blocks; ++ib) {
        int scaled_dot = 0;
        for (int group = 0; group < QK_K / 32; ++group) {
            int scale = 0;
            int minimum = 0;
            kimi_dense_stream_q4k_scale_min(x[ib], group, &scale, &minimum);
            int dot = 0;
            for (int offset = 0; offset < 4; ++offset) {
                const int position = 4 * lane + offset;
                const uint8_t packed = x[ib].qs[32 * (group / 2) + position];
                const int q4 = group & 1 ? packed >> 4 : packed & 0x0f;
                dot += q4 * (int) input[ib].qs[32 * group + position];
            }
            scaled_dot += scale * dot;
        }
        const float d = input[ib].d * __half2float(__low2half(x[ib].dm));
        sum = fmaf(d, (float) scaled_dot, sum);
        if (lane < 4) {
            const int group0 = 2 * lane;
            const int group1 = group0 + 1;
            int scale = 0;
            int minimum0 = 0;
            kimi_dense_stream_q4k_scale_min(x[ib], group0, &scale, &minimum0);
            int minimum1 = 0;
            kimi_dense_stream_q4k_scale_min(x[ib], group1, &scale, &minimum1);
            const int minimum_dot =
                minimum0 * ((int) input[ib].bsums[2 * group0] + (int) input[ib].bsums[2 * group0 + 1]) +
                minimum1 * ((int) input[ib].bsums[2 * group1] + (int) input[ib].bsums[2 * group1 + 1]);
            const float dmin = -input[ib].d * __half2float(__high2half(x[ib].dm));
            minimum_sum = fmaf(dmin, (float) minimum_dot, minimum_sum);
        }
    }
    const float sum0 = __shfl_sync(0x000000ffU, sum, 4) + sum;
    const float sum1 = __shfl_sync(0x000000ffU, sum, 5) +
        __shfl_sync(0x000000ffU, sum, 1);
    const float sum2 = __shfl_sync(0x000000ffU, sum, 6) +
        __shfl_sync(0x000000ffU, sum, 2);
    const float sum3 = __shfl_sync(0x000000ffU, sum, 7) +
        __shfl_sync(0x000000ffU, sum, 3);
    float minimum_total = 0.0f;
    if (lane < 4) {
        const float minimum0 = minimum_sum + __shfl_sync(0x0000000fU, minimum_sum, 2);
        const float minimum1 = __shfl_sync(0x0000000fU, minimum_sum, 1) +
            __shfl_sync(0x0000000fU, minimum_sum, 3);
        minimum_total = minimum0 + minimum1;
    }
    if (lane == 0) {
        output[column * output_rows + row] = ((sum0 + sum2) + (sum1 + sum3)) + minimum_total;
    }
}

static __global__ void kimi_dense_stream_q8_0_q8_0(
        const void * weights,
        size_t row_stride,
        const block_q8_0 * input,
        int64_t ne00,
        size_t rows,
        size_t output_rows,
        float * output) {
    const size_t row = blockIdx.x;
    const size_t column = blockIdx.y;
    const int lane = threadIdx.x;
    if (row >= rows || lane >= 8) {
        return;
    }

    const block_q8_0 * x =
        (const block_q8_0 *) ((const char *) weights + row * row_stride);
    const int blocks = (int) (ne00 / QK8_0);
    input += column * blocks;
    float sum = 0.0f;
    for (int ib = 0; ib < blocks; ++ib) {
        int dot = 0;
        for (int offset = 0; offset < 4; ++offset) {
            const int position = 4 * lane + offset;
            dot += (int) x[ib].qs[position] * (int) input[ib].qs[position];
        }
        const float d = __half2float(x[ib].d) * __half2float(input[ib].d);
        sum = fmaf(d, (float) dot, sum);
    }
    const float sum0 = __shfl_sync(0x000000ffU, sum, 4) + sum;
    const float sum1 = __shfl_sync(0x000000ffU, sum, 5) +
        __shfl_sync(0x000000ffU, sum, 1);
    const float sum2 = __shfl_sync(0x000000ffU, sum, 6) +
        __shfl_sync(0x000000ffU, sum, 2);
    const float sum3 = __shfl_sync(0x000000ffU, sum, 7) +
        __shfl_sync(0x000000ffU, sum, 3);
    if (lane == 0) {
        output[column * output_rows + row] = (sum0 + sum2) + (sum1 + sum3);
    }
}

__global__ void kimi_dense_stream_f32_to_bf16(const float * input, nv_bfloat16 * output, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) { output[i] = __float2bfloat16_rn(input[i]); }
}

static bool kimi_dense_stream_type_supported(ggml_type type) {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q8_0 ||
        type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

static bool kimi_dense_stream_elapsed(cudaEvent_t start, cudaEvent_t end, double * elapsed_ms) {
    float value = 0.0f;
    if (cudaEventElapsedTime(&value, start, end) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    *elapsed_ms = value;
    return true;
}

static bool kimi_dense_stream_pipeline_ensure_buffers(kimi_dense_stream_state & state) {
    for (size_t i = 0; i < 2; ++i) {
        if (!kimi_dense_stream_ensure_device(
                    &state.pipeline_buffers[i].device,
                    &state.pipeline_buffers[i].capacity,
                    state.pipeline_chunk_bytes)) {
            std::fprintf(
                stderr,
                "[kimi_dense_stream] pipeline buffer %llu allocation failed; using CPU fallback\n",
                (unsigned long long) i);
            return false;
        }
    }
    return true;
}

static void kimi_dense_stream_pipeline_note_tensor(
        kimi_dense_stream_state & state,
        const char * name,
        int type,
        const void * data,
        size_t bytes,
        int64_t ne00,
        int64_t ne01,
        size_t nb01) {
    auto found = state.pipeline_tensors.find(name);
    if (found == state.pipeline_tensors.end()) {
        kimi_dense_stream_tensor_desc desc;
        desc.type = type;
        desc.data = data;
        desc.bytes = bytes;
        desc.ne00 = ne00;
        desc.ne01 = ne01;
        desc.nb01 = nb01;
        state.pipeline_tensors.emplace(name, desc);
        state.pipeline_order.emplace_back(name);
    } else {
        found->second.type = type;
        found->second.data = data;
        found->second.bytes = bytes;
        found->second.ne00 = ne00;
        found->second.ne01 = ne01;
        found->second.nb01 = nb01;
    }

    if (!state.pipeline_order_ready && std::strcmp(name, "output.weight") == 0) {
        state.pipeline_order_ready = true;
        std::fprintf(
            stderr,
            "[kimi_dense_stream] pipeline order ready: tensors=%llu first=%s last=%s\n",
            (unsigned long long) state.pipeline_order.size(),
            state.pipeline_order.front().c_str(),
            state.pipeline_order.back().c_str());
    }
}

static const kimi_dense_stream_tensor_desc * kimi_dense_stream_pipeline_next_tensor(
        kimi_dense_stream_state & state,
        const char * current_name,
        const std::string ** next_name) {
    if (!state.pipeline_order_ready || state.pipeline_order.empty()) {
        return nullptr;
    }
    for (size_t i = 0; i < state.pipeline_order.size(); ++i) {
        if (state.pipeline_order[i] != current_name) {
            continue;
        }
        const std::string & name = state.pipeline_order[(i + 1) % state.pipeline_order.size()];
        const auto found = state.pipeline_tensors.find(name);
        if (found == state.pipeline_tensors.end()) {
            return nullptr;
        }
        *next_name = &state.pipeline_order[(i + 1) % state.pipeline_order.size()];
        return &found->second;
    }
    return nullptr;
}

static size_t kimi_dense_stream_pipeline_rows(
        const kimi_dense_stream_state & state,
        const kimi_dense_stream_tensor_desc & desc,
        size_t row_begin) {
    if (desc.nb01 == 0 || desc.nb01 > state.pipeline_chunk_bytes || row_begin >= (size_t) desc.ne01) {
        return 0;
    }
    const size_t rows_capacity = std::max((size_t) 1, state.pipeline_chunk_bytes / desc.nb01);
    return std::min(rows_capacity, (size_t) desc.ne01 - row_begin);
}

static int kimi_dense_stream_pipeline_find_buffer(
        const kimi_dense_stream_state & state,
        const std::string & tensor,
        size_t row_begin,
        size_t rows) {
    for (int i = 0; i < 2; ++i) {
        const kimi_dense_stream_pipeline_buffer & buffer = state.pipeline_buffers[i];
        if (buffer.has_copy && buffer.tensor == tensor &&
                buffer.row_begin == row_begin && buffer.rows == rows) {
            return i;
        }
    }
    return -1;
}

static bool kimi_dense_stream_pipeline_schedule_copy(
        kimi_dense_stream_state & state,
        int buffer_index,
        const std::string & tensor,
        const kimi_dense_stream_tensor_desc & desc,
        size_t row_begin,
        size_t rows) {
    if (buffer_index < 0 || buffer_index >= 2 || rows == 0 || row_begin + rows > (size_t) desc.ne01) {
        return false;
    }
    const size_t bytes = rows * desc.nb01;
    kimi_dense_stream_pipeline_buffer & buffer = state.pipeline_buffers[buffer_index];
    if (bytes > buffer.capacity) {
        return false;
    }
    if (buffer.has_copy && buffer.tensor == tensor &&
            buffer.row_begin == row_begin && buffer.rows == rows) {
        return true;
    }

    const auto registration = state.registrations.find(desc.data);
    if (registration == state.registrations.end() || !registration->second.direct) {
        return false;
    }
    if ((row_begin + rows) * desc.nb01 > desc.bytes) {
        return false;
    }

    auto check = [&](cudaError_t error, const char * operation) {
        if (error == cudaSuccess) {
            return true;
        }
        std::fprintf(
            stderr,
            "[kimi_dense_stream] pipeline copy %s failed: tensor=%s row=%llu rows=%llu buffer=%d "
            "cuda_error=%d (%s)\n",
            operation,
            tensor.c_str(),
            (unsigned long long) row_begin,
            (unsigned long long) rows,
            buffer_index,
            (int) error,
            cudaGetErrorString(error));
        return false;
    };

    if (buffer.has_compute &&
            !check(cudaStreamWaitEvent(state.transfer_stream, buffer.compute_done, 0), "wait-compute")) {
        return false;
    }
    if (!check(cudaEventRecord(buffer.copy_start, state.transfer_stream), "record-start")) {
        return false;
    }
    const uintptr_t source_begin = (uintptr_t) desc.data + row_begin * desc.nb01;
    const uintptr_t source_end = source_begin + bytes;
    if (registration->second.already_pinned) {
        if (!check(
                cudaMemcpyAsync(
                    buffer.device,
                    (const void *) source_begin,
                    bytes,
                    cudaMemcpyHostToDevice,
                    state.transfer_stream),
                "memcpy")) {
            return false;
        }
    } else {
        uintptr_t cursor = source_begin;
        for (const kimi_dense_stream_registered_range & range : state.registered_ranges) {
            if (range.end <= cursor) {
                continue;
            }
            if (range.begin > cursor || cursor >= source_end) {
                break;
            }
            const uintptr_t copy_end = std::min(range.end, source_end);
            if (!check(
                    cudaMemcpyAsync(
                        (char *) buffer.device + (cursor - source_begin),
                        (const void *) cursor,
                        copy_end - cursor,
                        cudaMemcpyHostToDevice,
                        state.transfer_stream),
                    "memcpy-segment")) {
                return false;
            }
            cursor = copy_end;
        }
        if (cursor != source_end) {
            std::fprintf(
                stderr,
                "[kimi_dense_stream] pipeline copy registration gap: tensor=%s source=%p bytes=%llu "
                "covered=%llu\n",
                tensor.c_str(),
                (const void *) source_begin,
                (unsigned long long) bytes,
                (unsigned long long) (cursor - source_begin));
            return false;
        }
    }
    if (!check(cudaEventRecord(buffer.copy_done, state.transfer_stream), "record-done")) {
        return false;
    }

    buffer.tensor = tensor;
    buffer.row_begin = row_begin;
    buffer.rows = rows;
    buffer.bytes = bytes;
    buffer.has_copy = true;
    buffer.has_compute = false;
    return true;
}

static bool kimi_dense_stream_pipeline_wait_copy(
        kimi_dense_stream_pipeline_buffer & buffer,
        double * h2d_ms,
        double * exposed_ms) {
    const cudaError_t query = cudaEventQuery(buffer.copy_done);
    if (query == cudaErrorNotReady) {
        const auto wait_start = std::chrono::steady_clock::now();
        if (cudaEventSynchronize(buffer.copy_done) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        const auto wait_done = std::chrono::steady_clock::now();
        *exposed_ms += std::chrono::duration<double, std::milli>(wait_done - wait_start).count();
    } else if (query != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    double copy_ms = 0.0;
    if (!kimi_dense_stream_elapsed(buffer.copy_start, buffer.copy_done, &copy_ms)) {
        return false;
    }
    *h2d_ms += copy_ms;
    return true;
}

// A tensor may need more chunks than the original fixed event array allowed.
// Keep timing events for its actual geometry; payload buffers remain bounded.
static bool kimi_dense_stream_pipeline_ensure_events(kimi_dense_stream_state & state, size_t chunks) {
    try {
        state.pipeline_compute_events.resize(std::max(chunks, state.pipeline_compute_events.size()));
    } catch (const std::bad_alloc &) {
        return false;
    }
    for (size_t i = 0; i < chunks; ++i) {
        auto & events = state.pipeline_compute_events[i];
        if ((!events.start && cudaEventCreate(&events.start) != cudaSuccess) ||
                (!events.done && cudaEventCreate(&events.done) != cudaSuccess)) {
            cudaGetLastError();
            return false;
        }
    }
    return true;
}

static bool kimi_dense_stream_pipeline_launch(
        kimi_dense_stream_state & state,
        kimi_dense_stream_pipeline_buffer & buffer,
        ggml_type type,
        int64_t ne00,
        int64_t ne01,
        int64_t ne11,
        size_t nb01,
        size_t row_begin,
        size_t rows,
        int64_t input_padded,
        size_t compute_event,
        double * launch_ms) {
    if (compute_event >= state.pipeline_compute_events.size()) {
        return false;
    }
    if (cudaEventRecord(state.pipeline_compute_events[compute_event].start, state.stream) != cudaSuccess) {
        return false;
    }

    const auto launch_start = std::chrono::steady_clock::now();
    bool ok = true;
    if (type == GGML_TYPE_Q4_K) {
        const dim3 grid((unsigned int) rows, (unsigned int) ne11);
        kimi_dense_stream_q4k_q8k<<<grid, 32, 0, state.stream>>>(
            buffer.device,
            nb01,
            (const block_q8_K *) state.device_input_q8,
            ne00,
            rows,
            (size_t) ne01,
            state.device_output + row_begin);
        ok = cudaGetLastError() == cudaSuccess;
    } else if (type == GGML_TYPE_Q8_0) {
        const dim3 grid((unsigned int) rows, (unsigned int) ne11);
        kimi_dense_stream_q8_0_q8_0<<<grid, 32, 0, state.stream>>>(
            buffer.device,
            nb01,
            (const block_q8_0 *) state.device_input_q8,
            ne00,
            rows,
            (size_t) ne01,
            state.device_output + row_begin);
        ok = cudaGetLastError() == cudaSuccess;
    } else {
        const float * weight_f32 = (const float *) buffer.device;
        int64_t leading_dimension = nb01 / sizeof(float);
        if (type != GGML_TYPE_F32) {
            const to_fp32_cuda_t dequantize = ggml_get_to_fp32_cuda(type);
            if (!dequantize) {
                return false;
            }
            dequantize(
                buffer.device,
                state.device_weight_f32,
                (int64_t) rows * ne00,
                state.stream);
            ok = cudaGetLastError() == cudaSuccess;
            weight_f32 = state.device_weight_f32;
            leading_dimension = ne00;
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;
        if (ok) {
            ok = cublasSgemm(
                state.cublas,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                (int) rows,
                (int) ne11,
                (int) ne00,
                &alpha,
                weight_f32,
                (int) leading_dimension,
                state.device_input,
                (int) ne00,
                &beta,
                state.device_output + row_begin,
                (int) ne01) == CUBLAS_STATUS_SUCCESS;
        }
    }
    const auto launch_done = std::chrono::steady_clock::now();
    *launch_ms += std::chrono::duration<double, std::milli>(launch_done - launch_start).count();
    if (!ok || cudaEventRecord(state.pipeline_compute_events[compute_event].done, state.stream) != cudaSuccess ||
            cudaEventRecord(buffer.compute_done, state.stream) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    buffer.has_compute = true;
    GGML_UNUSED(input_padded);
    return true;
}

static bool kimi_dense_stream_mul_mat_pipeline(
        kimi_dense_stream_state & state,
        ggml_type type,
        const char * src0_name,
        const void * src0_data,
        size_t src0_bytes,
        int64_t ne00,
        int64_t ne01,
        int64_t ne02,
        int64_t ne03,
        size_t nb01,
        const float * src1_data,
        int64_t ne11,
        int64_t ne12,
        int64_t ne13,
        size_t nb11,
        size_t nb12,
        size_t nb13,
        float * dst_data,
        size_t nb1,
        size_t nb2,
        size_t nb3) {
    kimi_dense_stream_pipeline_note_tensor(
        state, src0_name, (int) type, src0_data, src0_bytes, ne00, ne01, nb01);
    if (!state.pipeline_order_ready) {
        return false;
    }

    if (state.debug) {
        std::fprintf(
            stderr,
            "[kimi_dense_stream] debug: enter tensor=%s type=%d ne00=%lld ne01=%lld nb01=%llu bytes=%llu\n",
            src0_name,
            (int) type,
            (long long) ne00,
            (long long) ne01,
            (unsigned long long) nb01,
            (unsigned long long) src0_bytes);
    }

    if (ne02 != 1 || ne03 != 1 || ne11 <= 0 || ne11 > state.pipeline_max_batch ||
            ne12 != 1 || ne13 != 1 ||
            nb01 == 0 || (size_t) ne01 * nb01 > src0_bytes) {
        std::fprintf(
            stderr,
            "[kimi_dense_stream] pipeline shape unsupported for %s; using CPU fallback\n",
            src0_name);
        return false;
    }

    const auto total_start = std::chrono::steady_clock::now();
    const size_t input_bytes = (size_t) ne00 * (size_t) ne11 * sizeof(float);
    const size_t output_bytes = (size_t) ne01 * (size_t) ne11 * sizeof(float);
    const int64_t input_padded = GGML_PAD(ne00, MATRIX_ROW_PADDING);
    const kimi_dense_stream_tensor_desc & current = state.pipeline_tensors[src0_name];
    const size_t chunk_rows = kimi_dense_stream_pipeline_rows(state, current, 0);
    if (!chunk_rows || !kimi_dense_stream_pipeline_ensure_events(
            state, (size_t(ne01) - 1) / chunk_rows + 1)) {
        return false;
    }
    const size_t input_quantized_row_bytes = type == GGML_TYPE_Q4_K ?
        (size_t) (ne00 / QK_K) * sizeof(block_q8_K) :
        type == GGML_TYPE_Q8_0 ? (size_t) (ne00 / QK8_0) * sizeof(block_q8_0) :
        type == GGML_TYPE_F16 ? (size_t) ne00 * sizeof(ggml_fp16_t) : 0;
    const size_t input_quantized_bytes = input_quantized_row_bytes * (size_t) ne11;
    const size_t dequant_bytes = (type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) ?
        chunk_rows * (size_t) ne00 * sizeof(float) : 0;

    if (!kimi_dense_stream_pipeline_ensure_buffers(state) ||
            !kimi_dense_stream_ensure_pinned((void **) &state.host_input, &state.host_input_size, input_bytes) ||
            (input_quantized_bytes > 0 &&
                !kimi_dense_stream_ensure_pinned(
                    &state.host_input_quantized,
                    &state.host_input_quantized_size,
                    input_quantized_bytes)) ||
            !kimi_dense_stream_ensure_pinned((void **) &state.host_output, &state.host_output_size, output_bytes) ||
            !kimi_dense_stream_ensure_device((void **) &state.device_input, &state.device_input_size, input_bytes) ||
            ((type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q8_0) &&
                !kimi_dense_stream_ensure_device(
                    &state.device_input_q8,
                    &state.device_input_q8_size,
                    input_quantized_bytes)) ||
            !kimi_dense_stream_ensure_device((void **) &state.device_output, &state.device_output_size, output_bytes) ||
            (dequant_bytes > 0 &&
                !kimi_dense_stream_ensure_device(
                    (void **) &state.device_weight_f32,
                    &state.device_weight_f32_size,
                    dequant_bytes))) {
        std::fprintf(stderr, "[kimi_dense_stream] pipeline allocation failed for %s; using CPU fallback\n", src0_name);
        return false;
    }
    kimi_dense_stream_update_peaks(state);

    for (int64_t column = 0; column < ne11; ++column) {
        const float * source = (const float *) ((const char *) src1_data + (size_t) column * nb11);
        float * host_input = state.host_input + (size_t) column * ne00;
        std::memcpy(host_input, source, (size_t) ne00 * sizeof(float));
        if (type == GGML_TYPE_Q4_K) {
            block_q8_K * quantized = (block_q8_K *)
                ((char *) state.host_input_quantized + (size_t) column * input_quantized_row_bytes);
            quantize_row_q8_K_ref(source, quantized, ne00);
        } else if (type == GGML_TYPE_Q8_0) {
            block_q8_0 * quantized = (block_q8_0 *)
                ((char *) state.host_input_quantized + (size_t) column * input_quantized_row_bytes);
            quantize_row_q8_0_ref(source, quantized, ne00);
        } else if (type == GGML_TYPE_F16) {
            ggml_fp16_t * quantized = (ggml_fp16_t *)
                ((char *) state.host_input_quantized + (size_t) column * input_quantized_row_bytes);
            ggml_fp32_to_fp16_row(source, quantized, ne00);
            ggml_fp16_to_fp32_row(quantized, host_input, ne00);
        }
    }
    if (cudaEventRecord(state.event_weight_start, state.stream) != cudaSuccess ||
            cudaMemcpyAsync(
                state.device_input,
                state.host_input,
                input_bytes,
                cudaMemcpyHostToDevice,
                state.stream) != cudaSuccess ||
            ((type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q8_0) &&
                cudaMemcpyAsync(
                    state.device_input_q8,
                    state.host_input_quantized,
                    input_quantized_bytes,
                    cudaMemcpyHostToDevice,
                    state.stream) != cudaSuccess) ||
            cudaEventRecord(state.event_input_done, state.stream) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    size_t compute_events = 0;
    size_t prefetch_hits = 0;
    double weight_h2d_ms = 0.0;
    double weight_exposed_ms = 0.0;
    double launch_ms = 0.0;
    size_t row_begin = 0;
    while (row_begin < (size_t) ne01) {
        const size_t rows = kimi_dense_stream_pipeline_rows(state, current, row_begin);
        if (rows == 0 || compute_events >= state.pipeline_compute_events.size()) {
            return false;
        }

        int buffer_index = kimi_dense_stream_pipeline_find_buffer(state, src0_name, row_begin, rows);
        const bool prefetched = buffer_index >= 0;
        if (buffer_index >= 0) {
            ++prefetch_hits;
        } else {
            buffer_index = (int) (compute_events % 2);
            if (!kimi_dense_stream_pipeline_schedule_copy(
                        state, buffer_index, src0_name, current, row_begin, rows)) {
                std::fprintf(
                    stderr,
                    "[kimi_dense_stream] pipeline demand copy failed: tensor=%s chunk=%llu buffer=%d\n",
                    src0_name,
                    (unsigned long long) compute_events,
                    buffer_index);
                return false;
            }
        }
        if (state.debug) {
            std::fprintf(
                stderr,
                "[kimi_dense_stream] debug: demand tensor=%s chunk=%llu buffer=%d row=%llu rows=%llu prefetched=%d\n",
                src0_name,
                (unsigned long long) compute_events,
                buffer_index,
                (unsigned long long) row_begin,
                (unsigned long long) rows,
                prefetched ? 1 : 0);
        }

        const size_t next_row = row_begin + rows;
        const kimi_dense_stream_tensor_desc * next = nullptr;
        const std::string * next_name = nullptr;
        std::string current_name(src0_name);
        if (next_row < (size_t) ne01) {
            next = &current;
            next_name = &current_name;
        } else {
            next = kimi_dense_stream_pipeline_next_tensor(state, src0_name, &next_name);
        }
        if (next && next_name) {
            const size_t next_row_begin = next == &current ? next_row : 0;
            const size_t next_rows = kimi_dense_stream_pipeline_rows(state, *next, next_row_begin);
            const int next_buffer = 1 - buffer_index;
            if (next_rows > 0 &&
                    kimi_dense_stream_pipeline_find_buffer(
                        state, *next_name, next_row_begin, next_rows) < 0) {
                kimi_dense_stream_pipeline_schedule_copy(
                    state, next_buffer, *next_name, *next, next_row_begin, next_rows);
            }
        }

        kimi_dense_stream_pipeline_buffer & buffer = state.pipeline_buffers[buffer_index];
        if (!kimi_dense_stream_pipeline_wait_copy(buffer, &weight_h2d_ms, &weight_exposed_ms)) {
            std::fprintf(
                stderr,
                "[kimi_dense_stream] pipeline copy wait failed: tensor=%s chunk=%llu\n",
                src0_name,
                (unsigned long long) compute_events);
            return false;
        }
        if (state.debug) {
            std::fprintf(
                stderr,
                "[kimi_dense_stream] debug: copy-ready tensor=%s chunk=%llu\n",
                src0_name,
                (unsigned long long) compute_events);
        }
        if (!kimi_dense_stream_pipeline_launch(
                    state,
                    buffer,
                    type,
                    ne00,
                    ne01,
                    ne11,
                    nb01,
                    row_begin,
                    rows,
                    input_padded,
                    compute_events,
                    &launch_ms)) {
            std::fprintf(
                stderr,
                "[kimi_dense_stream] pipeline launch failed: tensor=%s chunk=%llu\n",
                src0_name,
                (unsigned long long) compute_events);
            return false;
        }
        if (state.debug) {
            std::fprintf(
                stderr,
                "[kimi_dense_stream] debug: launched tensor=%s chunk=%llu\n",
                src0_name,
                (unsigned long long) compute_events);
        }
        ++compute_events;
        row_begin = next_row;
    }

    if (state.debug) {
        std::fprintf(stderr, "[kimi_dense_stream] debug: output-sync-begin tensor=%s\n", src0_name);
    }
    if (cudaEventRecord(state.event_compute_done, state.stream) != cudaSuccess ||
            cudaMemcpyAsync(
                state.host_output,
                state.device_output,
                output_bytes,
                cudaMemcpyDeviceToHost,
                state.stream) != cudaSuccess ||
            cudaEventRecord(state.event_output_done, state.stream) != cudaSuccess ||
            cudaEventSynchronize(state.event_output_done) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (state.debug) {
        std::fprintf(stderr, "[kimi_dense_stream] debug: output-sync-done tensor=%s\n", src0_name);
    }

    double input_h2d_ms = 0.0;
    double compute_ms = 0.0;
    double output_d2h_ms = 0.0;
    if (!kimi_dense_stream_elapsed(state.event_weight_start, state.event_input_done, &input_h2d_ms) ||
            !kimi_dense_stream_elapsed(state.event_compute_done, state.event_output_done, &output_d2h_ms)) {
        return false;
    }
    for (size_t i = 0; i < compute_events; ++i) {
        double elapsed_ms = 0.0;
        if (!kimi_dense_stream_elapsed(
                    state.pipeline_compute_events[i].start, state.pipeline_compute_events[i].done, &elapsed_ms)) {
            return false;
        }
        compute_ms += elapsed_ms;
    }

    const auto scatter_start = std::chrono::steady_clock::now();
    for (int64_t column = 0; column < ne11; ++column) {
        std::memcpy(
            (char *) dst_data + (size_t) column * nb1,
            state.host_output + (size_t) column * ne01,
            (size_t) ne01 * sizeof(float));
    }
    const auto scatter_done = std::chrono::steady_clock::now();
    const auto total_done = std::chrono::steady_clock::now();

    kimi_dense_stream_profile & profile = state.profiles[src0_name];
    profile.calls += 1;
    profile.chunks += compute_events;
    profile.prefetch_hits += prefetch_hits;
    profile.weight_bytes += src0_bytes;
    profile.direct_weight_calls += 1;
    profile.direct_weight_bytes += src0_bytes;
    profile.weight_h2d_ms += weight_h2d_ms;
    profile.weight_exposed_ms += weight_exposed_ms;
    profile.input_h2d_ms += input_h2d_ms;
    profile.compute_ms += compute_ms;
    profile.launch_ms += launch_ms;
    profile.output_d2h_ms += output_d2h_ms;
    profile.scatter_ms += std::chrono::duration<double, std::milli>(scatter_done - scatter_start).count();
    profile.total_ms += std::chrono::duration<double, std::milli>(total_done - total_start).count();

    if (state.debug) {
        std::fprintf(
            stderr,
            "[kimi_dense_stream] debug: exit tensor=%s chunks=%llu h2d_ms=%.3f exposed_ms=%.3f compute_ms=%.3f\n",
            src0_name,
            (unsigned long long) compute_events,
            weight_h2d_ms,
            weight_exposed_ms,
            compute_ms);
    }

    GGML_UNUSED(nb11);
    GGML_UNUSED(nb12);
    GGML_UNUSED(nb13);
    GGML_UNUSED(nb1);
    GGML_UNUSED(nb2);
    GGML_UNUSED(nb3);
    return true;
}

} // namespace

extern "C" bool ggml_cuda_kimi_dense_stream_should_intercept(const char * tensor_name) {
    static const bool pipeline = kimi_dense_stream_env_enabled("GGML_KIMI_DENSE_STREAM_PIPELINE");
    if (!pipeline || (tensor_name && std::strcmp(tensor_name, "output.weight") == 0)) {
        return true;
    }
    return kimi_dense_stream_get_state().pipeline_started.load(std::memory_order_acquire);
}

extern "C" bool ggml_cuda_kimi_dense_stream_mul_mat(
    int src0_type_int,
    const char * src0_name,
    const void * src0_data,
    size_t src0_bytes,
    int64_t ne00,
    int64_t ne01,
    int64_t ne02,
    int64_t ne03,
    size_t nb01,
    size_t nb02,
    size_t nb03,
    const float * src1_data,
    int64_t ne11,
    int64_t ne12,
    int64_t ne13,
    size_t nb11,
    size_t nb12,
    size_t nb13,
    float * dst_data,
    size_t nb1,
    size_t nb2,
    size_t nb3) {

    const ggml_type type = (ggml_type) src0_type_int;
    if (!src0_name || !src0_data || !src1_data || !dst_data ||
            !kimi_dense_stream_type_supported(type) ||
            ne00 <= 0 || ne01 <= 0 || ne02 <= 0 || ne03 <= 0 ||
            ne11 <= 0 || ne12 <= 0 || ne13 <= 0 ||
            ne12 % ne02 != 0 || ne13 % ne03 != 0) {
        return false;
    }

    const int64_t vectors = ne11 * ne12 * ne13;
    const size_t input_elements = (size_t) vectors * (size_t) ne00;
    const size_t output_elements = (size_t) vectors * (size_t) ne01;
    const size_t input_bytes = input_elements * sizeof(float);
    const size_t output_bytes = output_elements * sizeof(float);
    const int64_t input_padded = GGML_PAD(ne00, MATRIX_ROW_PADDING);
    const size_t input_q8_bytes =
        (size_t) vectors * (size_t) (input_padded / QK8_1) * sizeof(block_q8_1);

    kimi_dense_stream_state & state = kimi_dense_stream_get_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!kimi_dense_stream_init(state)) {
        return false;
    }
    if (state.pipeline && std::strcmp(src0_name, "output.weight") == 0) {
        state.pipeline_started.store(true, std::memory_order_release);
    }

    const auto total_start = std::chrono::steady_clock::now();
    const bool zero_copy_target = state.zero_copy &&
        (state.zero_copy_tensor.empty() || state.zero_copy_tensor == src0_name);
    const bool registered_weight = zero_copy_target &&
        kimi_dense_stream_register_weight(state, src0_name, src0_data, src0_bytes);
    const bool direct_weight = registered_weight && !state.zero_copy_register_only;
    if (state.pipeline) {
        if (!direct_weight) {
            std::fprintf(
                stderr,
                "[kimi_dense_stream] pipeline requires registered weight %s; using CPU fallback\n",
                src0_name);
            return false;
        }
        return kimi_dense_stream_mul_mat_pipeline(
            state,
            type,
            src0_name,
            src0_data,
            src0_bytes,
            ne00,
            ne01,
            ne02,
            ne03,
            nb01,
            src1_data,
            ne11,
            ne12,
            ne13,
            nb11,
            nb12,
            nb13,
            dst_data,
            nb1,
            nb2,
            nb3);
    }
    if ((!direct_weight &&
                !kimi_dense_stream_ensure_pinned(&state.host_weight, &state.host_weight_size, src0_bytes)) ||
            !kimi_dense_stream_ensure_pinned((void **) &state.host_input, &state.host_input_size, input_bytes) ||
            !kimi_dense_stream_ensure_pinned((void **) &state.host_output, &state.host_output_size, output_bytes) ||
            !kimi_dense_stream_ensure_device(&state.device_weight, &state.device_weight_size, src0_bytes) ||
            !kimi_dense_stream_ensure_device((void **) &state.device_input, &state.device_input_size, input_bytes) ||
            !kimi_dense_stream_ensure_device((void **) &state.device_output, &state.device_output_size, output_bytes) ||
            ((type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q8_0) &&
                !kimi_dense_stream_ensure_device(&state.device_input_q8, &state.device_input_q8_size, input_q8_bytes)) ||
            ((type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) &&
                !kimi_dense_stream_ensure_device((void **) &state.device_input_f16, &state.device_input_f16_size,
                    input_elements * sizeof(__half)))) {
        std::fprintf(stderr, "[kimi_dense_stream] staging allocation failed for %s; using CPU fallback\n", src0_name);
        return false;
    }
    kimi_dense_stream_update_peaks(state);

    const auto stage_start = std::chrono::steady_clock::now();
    if (!direct_weight) {
        std::memcpy(state.host_weight, src0_data, src0_bytes);
    }
    const auto stage_done = std::chrono::steady_clock::now();

    size_t vector_index = 0;
    for (int64_t i13 = 0; i13 < ne13; ++i13) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            for (int64_t i11 = 0; i11 < ne11; ++i11, ++vector_index) {
                const char * src = (const char *) src1_data + i11 * nb11 + i12 * nb12 + i13 * nb13;
                std::memcpy(state.host_input + vector_index * ne00, src, (size_t) ne00 * sizeof(float));
            }
        }
    }

    if (cudaEventRecord(state.event_weight_start, state.stream) != cudaSuccess ||
            cudaMemcpyAsync(state.device_weight, direct_weight ? src0_data : state.host_weight, src0_bytes,
                cudaMemcpyHostToDevice, state.stream) != cudaSuccess ||
            cudaEventRecord(state.event_weight_done, state.stream) != cudaSuccess ||
            cudaMemcpyAsync(state.device_input, state.host_input, input_bytes,
                cudaMemcpyHostToDevice, state.stream) != cudaSuccess ||
            cudaEventRecord(state.event_input_done, state.stream) != cudaSuccess) {
        cudaGetLastError();
        cudaStreamSynchronize(state.stream);
        return false;
    }

    if (type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) {
        const int threads = 256;
        const int blocks = (int) ((input_elements + threads - 1) / threads);
        if (type == GGML_TYPE_BF16) {
            kimi_dense_stream_f32_to_bf16<<<blocks, threads, 0, state.stream>>>(
                state.device_input, reinterpret_cast<nv_bfloat16 *>(state.device_input_f16), input_elements);
        } else {
            kimi_dense_stream_f32_to_f16<<<blocks, threads, 0, state.stream>>>(
                state.device_input, state.device_input_f16, input_elements);
        }
        if (cudaGetLastError() != cudaSuccess) {
            cudaStreamSynchronize(state.stream);
            return false;
        }
    }

    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;
    vector_index = 0;
    for (int64_t i13 = 0; i13 < ne13; ++i13) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            const int64_t i03 = i13 / r3;
            const int64_t i02 = i12 / r2;
            const char * weight = (const char *) state.device_weight + i02 * nb02 + i03 * nb03;
            const int64_t columns_per_call = (type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) ? ne11 : 1;
            for (int64_t i11 = 0; i11 < ne11; i11 += columns_per_call, vector_index += columns_per_call) {
                float * output = state.device_output + vector_index * ne01;
                bool ok = true;
                if (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q8_0) {
                    void * input_q8 = (char *) state.device_input_q8 +
                        vector_index * (size_t) (input_padded / QK8_1) * sizeof(block_q8_1);
                    ok = ggml_cuda_moe_stream_mmvq_dev(
                        src0_type_int,
                        weight,
                        ne01,
                        ne00,
                        nb01,
                        state.device_input + vector_index * ne00,
                        input_q8,
                        output,
                        state.stream);
                } else if (type == GGML_TYPE_F32) {
                    const float alpha = 1.0f;
                    const float beta = 0.0f;
                    ok = cublasSgemv(
                        state.cublas,
                        CUBLAS_OP_T,
                        (int) ne00,
                        (int) ne01,
                        &alpha,
                        (const float *) weight,
                        (int) (nb01 / sizeof(float)),
                        state.device_input + vector_index * ne00,
                        1,
                        &beta,
                        output,
                        1) == CUBLAS_STATUS_SUCCESS;
                } else {
                    const float alpha = 1.0f;
                    const float beta = 0.0f;
                    ok = cublasGemmEx(
                        state.cublas,
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        (int) ne01,
                        (int) columns_per_call,
                        (int) ne00,
                        &alpha,
                        weight,
                        type == GGML_TYPE_BF16 ? CUDA_R_16BF : CUDA_R_16F,
                        (int) (nb01 / sizeof(__half)),
                        state.device_input_f16 + vector_index * ne00,
                        type == GGML_TYPE_BF16 ? CUDA_R_16BF : CUDA_R_16F,
                        (int) ne00,
                        &beta,
                        output,
                        CUDA_R_32F,
                        (int) ne01,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS;
                }
                if (!ok) {
                    cudaStreamSynchronize(state.stream);
                    return false;
                }
            }
        }
    }

    if (cudaEventRecord(state.event_compute_done, state.stream) != cudaSuccess ||
            cudaMemcpyAsync(state.host_output, state.device_output, output_bytes,
                cudaMemcpyDeviceToHost, state.stream) != cudaSuccess ||
            cudaEventRecord(state.event_output_done, state.stream) != cudaSuccess ||
            cudaEventSynchronize(state.event_output_done) != cudaSuccess) {
        cudaGetLastError();
        cudaStreamSynchronize(state.stream);
        return false;
    }

    const auto scatter_start = std::chrono::steady_clock::now();
    vector_index = 0;
    for (int64_t i13 = 0; i13 < ne13; ++i13) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            for (int64_t i11 = 0; i11 < ne11; ++i11, ++vector_index) {
                char * dst = (char *) dst_data + i11 * nb1 + i12 * nb2 + i13 * nb3;
                std::memcpy(dst, state.host_output + vector_index * ne01, (size_t) ne01 * sizeof(float));
            }
        }
    }
    const auto scatter_done = std::chrono::steady_clock::now();

    double weight_h2d_ms = 0.0;
    double input_h2d_ms = 0.0;
    double compute_ms = 0.0;
    double output_d2h_ms = 0.0;
    if (!kimi_dense_stream_elapsed(state.event_weight_start, state.event_weight_done, &weight_h2d_ms) ||
            !kimi_dense_stream_elapsed(state.event_weight_done, state.event_input_done, &input_h2d_ms) ||
            !kimi_dense_stream_elapsed(state.event_input_done, state.event_compute_done, &compute_ms) ||
            !kimi_dense_stream_elapsed(state.event_compute_done, state.event_output_done, &output_d2h_ms)) {
        return false;
    }

    const auto total_done = std::chrono::steady_clock::now();
    kimi_dense_stream_profile & profile = state.profiles[src0_name];
    profile.calls += 1;
    profile.weight_bytes += src0_bytes;
    profile.direct_weight_calls += direct_weight ? 1 : 0;
    profile.direct_weight_bytes += direct_weight ? src0_bytes : 0;
    profile.host_stage_ms += std::chrono::duration<double, std::milli>(stage_done - stage_start).count();
    profile.weight_h2d_ms += weight_h2d_ms;
    profile.input_h2d_ms += input_h2d_ms;
    profile.compute_ms += compute_ms;
    profile.output_d2h_ms += output_d2h_ms;
    profile.scatter_ms += std::chrono::duration<double, std::milli>(scatter_done - scatter_start).count();
    profile.total_ms += std::chrono::duration<double, std::milli>(total_done - total_start).count();
    return true;
}

#else

extern "C" bool ggml_cuda_kimi_dense_stream_should_intercept(const char *) {
    return true;
}

extern "C" bool ggml_cuda_kimi_dense_stream_mul_mat(
    int, const char *, const void *, size_t,
    int64_t, int64_t, int64_t, int64_t, size_t, size_t, size_t,
    const float *, int64_t, int64_t, int64_t, size_t, size_t, size_t,
    float *, size_t, size_t, size_t) {
    return false;
}

#endif
