#include "ggml-kimi-q3k-parity.h"
#define GGML_COMMON_DECL_CUDA
#include "ggml-common.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

namespace {
struct device_job {
    const char * weights;
    size_t row_stride, q8_offset, output_offset;
    int columns, rows;
    float scale;
    int apply_scale;
};

// Each group of eight lanes exactly follows AVX2's eight FP32 accumulators.
// The integer work in each 256-element block has no saturation or overflow
// for valid Q3_K and Q8_K blocks. Rounding is explicit at every FP operation.
__global__ void q3k_avx2(const device_job * jobs, const char * q8, float * output) {
    const device_job job = jobs[blockIdx.y];
    const int lane = threadIdx.x & 7;
    const int row = blockIdx.x * 16 + threadIdx.x / 8;
    if (row >= job.rows) return;
    const auto * x = reinterpret_cast<const block_q3_K *>(job.weights + row * job.row_stride);
    const auto * y = reinterpret_cast<const block_q8_K *>(q8 + job.q8_offset);
    float acc = 0.0f;
    for (int ib = 0; ib < job.columns / QK_K; ++ib) {
        int sumi = 0;
#pragma unroll
        for (int group = 0; group < 8; ++group) {
            const int sg = 2 * group + lane / 4;
            const int scale = ((x[ib].scales[sg & 7] >> (sg >= 8 ? 4 : 0)) & 15) |
                (((x[ib].scales[8 + (sg & 3)] >> (2 * (sg / 4))) & 3) << 4);
            int dot = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int p = 4 * lane + k;
                const int low = (x[ib].qs[(group / 4) * 32 + p] >> (2 * (group & 3))) & 3;
                const int quant = low - ((x[ib].hmask[p] & (1 << group)) ? 0 : 4);
                dot += quant * int(y[ib].qs[group * 32 + p]);
            }
            sumi += (scale - 32) * dot;
        }
        const float d = __fmul_rn(y[ib].d, __half2float(x[ib].d));
        acc = __fmaf_rn(d, __int2float_rn(sumi), acc);
    }
    const unsigned mask = 0xffu << (8 * ((threadIdx.x & 31) / 8));
    const float a = __fadd_rn(__shfl_sync(mask, acc, 4, 8), __shfl_sync(mask, acc, 0, 8));
    const float b = __fadd_rn(__shfl_sync(mask, acc, 6, 8), __shfl_sync(mask, acc, 2, 8));
    const float c = __fadd_rn(__shfl_sync(mask, acc, 5, 8), __shfl_sync(mask, acc, 1, 8));
    const float d = __fadd_rn(__shfl_sync(mask, acc, 7, 8), __shfl_sync(mask, acc, 3, 8));
    const float value = __fadd_rn(__fadd_rn(a, b), __fadd_rn(c, d));
    if (lane == 0) output[job.output_offset + row] = job.apply_scale ? __fmul_rn(value, job.scale) : value;
}

struct device_scope {
    int previous = -1;
    cudaError_t status;
    explicit device_scope(int target) {
        status = cudaGetDevice(&previous);
        if (status == cudaSuccess && previous != target) status = cudaSetDevice(target);
    }
    ~device_scope() { if (previous >= 0) cudaSetDevice(previous); }
};
}

struct kimi_q3k_gpu {
    kimi_q3k_gpu_config config{};
    cudaStream_t stream = nullptr;
    cudaEvent_t event = nullptr;
    device_job * h_jobs = nullptr, * d_jobs = nullptr;
    char * h_q8 = nullptr, * d_q8 = nullptr;
    float * h_output = nullptr, * d_output = nullptr;
    size_t job_bytes = 0, q8_bytes = 0, output_bytes = 0;
    int njobs = 0;
    bool busy = false, failed = false;
    char message[256]{};
    bool check(cudaError_t e, const char * what) {
        if (e == cudaSuccess) return true;
        std::snprintf(message, sizeof(message), "%s: %s", what, cudaGetErrorString(e));
        failed = true;
        return false;
    }
    int invalid(const char * why) { std::snprintf(message, sizeof(message), "%s", why); return 0; }
};

namespace {
int destroy(kimi_q3k_gpu * c) {
    if (!c) return 1;
    device_scope device(c->config.device);
    bool ok = device.status == cudaSuccess;
    if (ok) {
        if (c->stream) ok = cudaStreamSynchronize(c->stream) == cudaSuccess && ok;
        if (c->event) ok = cudaEventDestroy(c->event) == cudaSuccess && ok;
        if (c->d_jobs) ok = cudaFree(c->d_jobs) == cudaSuccess && ok;
        if (c->d_q8) ok = cudaFree(c->d_q8) == cudaSuccess && ok;
        if (c->d_output) ok = cudaFree(c->d_output) == cudaSuccess && ok;
        if (c->h_jobs) ok = cudaFreeHost(c->h_jobs) == cudaSuccess && ok;
        if (c->h_q8) ok = cudaFreeHost(c->h_q8) == cudaSuccess && ok;
        if (c->h_output) ok = cudaFreeHost(c->h_output) == cudaSuccess && ok;
        if (c->stream) ok = cudaStreamDestroy(c->stream) == cudaSuccess && ok;
    }
    delete c;
    return ok;
}

kimi_q3k_gpu * create(const kimi_q3k_gpu_config * config, char * error, size_t size) {
    auto fail = [&](const char * why) -> kimi_q3k_gpu * {
        if (error && size) std::snprintf(error, size, "%s", why);
        return nullptr;
    };
    const char * enabled = std::getenv("GGML_KIMI_Q3K_GPU_PARITY");
    if (!enabled || std::strcmp(enabled, "1")) return fail("GGML_KIMI_Q3K_GPU_PARITY must be 1");
    if (!config || config->device < 0 || config->max_jobs < 1 || config->max_jobs > 64 ||
            config->max_columns < 256 || config->max_columns > 32768 || config->max_columns % 256 ||
            config->max_rows < 1 || config->max_rows > 32768) return fail("invalid parity capacity");
    device_scope device(config->device);
    if (device.status != cudaSuccess) return fail(cudaGetErrorString(device.status));
    auto * c = new (std::nothrow) kimi_q3k_gpu;
    if (!c) return fail("parity handle allocation failed");
    c->config = *config;
    c->job_bytes = config->max_jobs * sizeof(device_job);
    c->q8_bytes = size_t(config->max_jobs) * (config->max_columns / 256) * sizeof(block_q8_K);
    c->output_bytes = size_t(config->max_jobs) * config->max_rows * sizeof(float);
    if (!c->check(cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking), "stream") ||
            !c->check(cudaEventCreateWithFlags(&c->event, cudaEventDisableTiming), "event") ||
            !c->check(cudaMallocHost(&c->h_jobs, c->job_bytes), "host jobs") ||
            !c->check(cudaMallocHost(&c->h_q8, c->q8_bytes), "host Q8_K") ||
            !c->check(cudaMallocHost(&c->h_output, c->output_bytes), "host output") ||
            !c->check(cudaMalloc(&c->d_jobs, c->job_bytes), "device jobs") ||
            !c->check(cudaMalloc(&c->d_q8, c->q8_bytes), "device Q8_K") ||
            !c->check(cudaMalloc(&c->d_output, c->output_bytes), "device output")) {
        fail(c->message); destroy(c); return nullptr;
    }
    if (error && size) error[0] = 0;
    return c;
}

int query(kimi_q3k_gpu * c) {
    if (!c || c->failed) return -1;
    if (!c->busy) return 1;
    device_scope device(c->config.device);
    if (!c->check(device.status, "query device")) return -1;
    const auto rc = cudaEventQuery(c->event);
    if (rc == cudaErrorNotReady) return 0;
    if (!c->check(rc, "query event")) return -1;
    c->busy = false;
    return 1;
}

int wait(kimi_q3k_gpu * c) {
    if (!c || c->failed) return 0;
    if (!c->busy) return 1;
    device_scope device(c->config.device);
    if (!c->check(device.status, "wait device") || !c->check(cudaEventSynchronize(c->event), "wait event")) return 0;
    c->busy = false;
    return 1;
}

int submit(kimi_q3k_gpu * c, const kimi_q3k_gpu_job * jobs, int n, const void * q8, size_t bytes) {
    if (!c || c->failed) return 0;
    if (c->busy) return c->invalid("consume the previous event before submitting");
    if (!jobs || !q8 || n < 1 || n > c->config.max_jobs || bytes > c->q8_bytes)
        return c->invalid("invalid jobs or Q8_K capacity");
    device_scope device(c->config.device);
    if (!c->check(device.status, "submit device")) return 0;
    size_t out = 0;
    int max_rows = 0;
    // Validate the entire batch before overwriting the previous result layout.
    for (int i = 0; i < n; ++i) {
        const auto & j = jobs[i];
        if (!j.weights || j.columns < 256 || j.columns > c->config.max_columns || j.columns % 256 ||
                j.rows < 1 || j.rows > c->config.max_rows || j.q8_offset % alignof(block_q8_K) ||
                j.q8_offset > bytes || size_t(j.columns / 256) * sizeof(block_q8_K) > bytes - j.q8_offset ||
                j.row_stride < size_t(j.columns / 256) * sizeof(block_q3_K) || j.row_stride % alignof(block_q3_K) ||
                j.weight_bytes < size_t(j.columns / 256) * sizeof(block_q3_K) ||
                (j.rows > 1 && j.row_stride > (j.weight_bytes - size_t(j.columns / 256) * sizeof(block_q3_K)) / (j.rows - 1)) ||
                (j.apply_scale != 0 && j.apply_scale != 1) || !std::isfinite(j.scale))
            return c->invalid("unsupported parity job layout/range");
        cudaPointerAttributes attr{};
        if (!c->check(cudaPointerGetAttributes(&attr, j.weights), "weight pointer")) return 0;
        if (attr.type != cudaMemoryTypeDevice || attr.device != c->config.device)
            return c->invalid("weights must reside on the handle device");
    }
    for (int i = 0; i < n; ++i) {
        const auto & j = jobs[i];
        c->h_jobs[i] = {static_cast<const char *>(j.weights), j.row_stride, j.q8_offset, out,
            j.columns, j.rows, j.scale, j.apply_scale};
        out += j.rows;
        max_rows = std::max(max_rows, j.rows);
    }
    c->njobs = n;
    std::memcpy(c->h_q8, q8, bytes);
    if (!c->check(cudaMemcpyAsync(c->d_jobs, c->h_jobs, n * sizeof(device_job), cudaMemcpyHostToDevice, c->stream), "jobs H2D") ||
            !c->check(cudaMemcpyAsync(c->d_q8, c->h_q8, bytes, cudaMemcpyHostToDevice, c->stream), "Q8_K H2D")) return 0;
    q3k_avx2<<<dim3((max_rows + 15) / 16, n), 128, 0, c->stream>>>(c->d_jobs, c->d_q8, c->d_output);
    if (!c->check(cudaGetLastError(), "Q3_K kernel") ||
            !c->check(cudaMemcpyAsync(c->h_output, c->d_output, out * sizeof(float), cudaMemcpyDeviceToHost, c->stream), "result D2H") ||
            !c->check(cudaEventRecord(c->event, c->stream), "completion event")) return 0;
    c->busy = true;
    return 1;
}

const float * result(kimi_q3k_gpu * c, int job, int * rows) {
    if (query(c) != 1 || job < 0 || job >= c->njobs) return nullptr;
    if (rows) *rows = c->h_jobs[job].rows;
    return c->h_output + c->h_jobs[job].output_offset;
}
const char * error(kimi_q3k_gpu * c) { return c ? c->message : "null parity handle"; }
size_t host_bytes(kimi_q3k_gpu * c) { return c ? c->job_bytes + c->q8_bytes + c->output_bytes : 0; }
size_t device_bytes(kimi_q3k_gpu * c) { return host_bytes(c); }
int upload(kimi_q3k_gpu * c, const void * src, size_t bytes, void ** dst) {
    if (!c || c->failed) return 0;
    if (c->busy || !src || !dst || !bytes) return c->invalid("invalid upload or handle busy");
    device_scope device(c->config.device);
    if (!c->check(device.status, "upload device")) return 0;
    *dst = nullptr;
    if (!c->check(cudaMalloc(dst, bytes), "weight allocation")) return 0;
    if (!c->check(cudaMemcpyAsync(*dst, src, bytes, cudaMemcpyHostToDevice, c->stream), "weight upload") ||
            !c->check(cudaStreamSynchronize(c->stream), "weight upload completion")) {
        cudaFree(*dst); *dst = nullptr; return 0;
    }
    return 1;
}
int release_weights(kimi_q3k_gpu * c, void * ptr) {
    if (!c || c->busy) return 0;
    device_scope device(c->config.device);
    return c->check(device.status, "release device") && c->check(cudaFree(ptr), "release weights");
}
const kimi_q3k_gpu_api_v1 api = {1, sizeof(kimi_q3k_gpu_job), create, submit, query, wait, result, error,
    destroy, host_bytes, device_bytes, upload, release_weights};
}

extern "C" KIMI_Q3K_EXPORT const kimi_q3k_gpu_api_v1 * kimi_q3k_gpu_get_api_v1() { return &api; }
