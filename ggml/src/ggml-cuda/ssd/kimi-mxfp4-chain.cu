// Optional GPU hidden chain for the MXFP4 VRAM tier (DeepSeek-V4.1 routed experts).
// Like kimi-q3k-chain.cu, this extension compiles the unchanged matvec parity source
// into the same DSO and adds one submission per layer: one H2D of the job descriptors
// and the Q8_0 input row, the stage-1 up/gate matvec, the clamped SwiGLU with the
// device expf, device Q8_0 of the hidden rows, the stage-2 down matvec reading that
// device buffer, and one D2H of the unweighted down rows. Not bitwise: expf, the device
// quantizer and (for the rows kernels) the matvec accumulation order differ from the
// CPU sink in the last bits. Selected by GGML_KIMI_CPU_SINK_VRAM_MXFP4_GPU_HIDDEN=1.
#include "kimi-mxfp4-parity.cu"
#include "ggml-kimi-mxfp4-chain.h"

namespace {
constexpr int chain_rows_per_block = 64;

// kvalues_mxfp4 without a switch: 0,1,2,3 for the low values, 4,6,8,12 above.
__device__ inline int mxfp4_value_fast(int nibble) {
    const int m = nibble & 7;
    const int magnitude = m < 4 ? m : (2 + (m & 1)) << ((m >> 1) - 1);
    return (nibble & 8) ? -magnitude : magnitude;
}

// One thread per output row: each block's 32 products in one int sum, then the block
// scale times that sum added in block order (ggml_vec_dot_mxfp4_q8_0_generic order).
__global__ void mxfp4_rows(const device_job * jobs, const char * q8, float * output) {
    const device_job job = jobs[blockIdx.y];
    const int row = blockIdx.x * chain_rows_per_block + threadIdx.x;
    if (row >= job.rows) return;
    const auto * x = reinterpret_cast<const block_mxfp4 *>(job.weights + row * job.row_stride);
    const auto * y = reinterpret_cast<const block_q8_0 *>(q8 + job.q8_offset);
    const int blocks = job.columns / QK_MXFP4;
    float sum = 0.0f;
    for (int ib = 0; ib < blocks; ++ib) {
        int dot = 0;
        for (int j = 0; j < 16; ++j) {
            dot += mxfp4_value_fast(x[ib].qs[j] & 15) * int(y[ib].qs[j]) +
                   mxfp4_value_fast(x[ib].qs[j] >> 4) * int(y[ib].qs[j + 16]);
        }
        sum += (__half2float(y[ib].d) * e8m0_half(x[ib].e)) * float(dot);
    }
    output[job.output_offset + row] = sum;
}

// Stage 2 on float hidden rows (q8_offset is a float index here): no hidden Q8_0.
__global__ void mxfp4_float_rows(const device_job * jobs, const float * hidden, float * output) {
    const device_job job = jobs[blockIdx.y];
    const int row = blockIdx.x * chain_rows_per_block + threadIdx.x;
    if (row >= job.rows) return;
    const auto * x = reinterpret_cast<const block_mxfp4 *>(job.weights + row * job.row_stride);
    const float * h = hidden + job.q8_offset;
    const int blocks = job.columns / QK_MXFP4;
    float sum = 0.0f;
    for (int ib = 0; ib < blocks; ++ib) {
        const float * hb = h + ib * QK_MXFP4;
        float dot = 0.0f;
        for (int j = 0; j < 16; ++j) {
            dot += float(mxfp4_value_fast(x[ib].qs[j] & 15)) * hb[j] + float(mxfp4_value_fast(x[ib].qs[j] >> 4)) * hb[j + 16];
        }
        sum += e8m0_half(x[ib].e) * dot;
    }
    output[job.output_offset + row] = sum;
}

// kimi-cpu-sink.cpp compute_rows, SWIGLU_CLAMP: silu(min(g, L)) * clamp(u, -L, L).
// Stage-1 output row layout: job 2*hit is up, job 2*hit+1 is gate.
__global__ void chain_swiglu(const float * up_gate, float * hidden, int hits, int rows, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hits * rows) return;
    const int hit = i / rows, row = i % rows;
    const float g = fminf(up_gate[(2 * hit + 1) * rows + row], limit);
    const float u = fminf(fmaxf(up_gate[2 * hit * rows + row], -limit), limit);
    hidden[i] = g / (1.0f + expf(-g)) * u;
}

__device__ inline float round_half_even(float v) {
    const float r = roundf(v);
    return fabsf(r - v) == 0.5f ? 2.0f * roundf(0.5f * v) : r;
}

// quantize_row_q8_0 as on AVX2: d = max|x| / 127 stored as FP16, q = x * (127 / max|x|)
// rounded half to even. One block of 32 threads per Q8_0 block of every hidden row.
__global__ void chain_q8_0(const float * x, block_q8_0 * y) {
    const int b = blockIdx.x, j = threadIdx.x;
    const float * v = x + b * QK8_0;
    __shared__ float id;
    if (j == 0) {
        float amax = 0.0f;
        for (int k = 0; k < QK8_0; ++k) amax = fmaxf(amax, fabsf(v[k]));
        id = amax != 0.0f ? 127.0f / amax : 0.0f;
        y[b].d = __float2half(amax / 127.0f);
    }
    __syncthreads();
    y[b].qs[j] = int8_t(int(round_half_even(v[j] * id)));
}
}

struct kimi_mxfp4_gpu_chain {
    kimi_mxfp4_gpu * base = nullptr;
    int latent = 0, hidden = 0, max_hits = 0, kernel = 0, hits = 0;
    bool keep_hidden = false;
    size_t input_offset = 0, upload_bytes = 0, hidden_float_bytes = 0, hidden_q8_bytes = 0, kept_bytes = 0;
    char * host_upload = nullptr, * device_upload = nullptr; // [2*max_hits up/gate jobs][max_hits down jobs][Q8_0 input]
    float * hidden_float = nullptr;
    char * hidden_q8 = nullptr;
    char * host_hidden = nullptr; // keep_hidden only
};

namespace {
int chain_destroy(kimi_mxfp4_gpu_chain * h) {
    if (!h) return 1;
    const auto c = h->base;
    device_scope device(c->config.device);
    bool ok = device.status == cudaSuccess;
    if (ok) {
        if (c->stream) ok = cudaStreamSynchronize(c->stream) == cudaSuccess && ok;
        if (h->host_upload) ok = cudaFreeHost(h->host_upload) == cudaSuccess && ok;
        if (h->device_upload) ok = cudaFree(h->device_upload) == cudaSuccess && ok;
        if (h->hidden_float) ok = cudaFree(h->hidden_float) == cudaSuccess && ok;
        if (h->hidden_q8) ok = cudaFree(h->hidden_q8) == cudaSuccess && ok;
        if (h->host_hidden) ok = cudaFreeHost(h->host_hidden) == cudaSuccess && ok;
    }
    delete h;
    return ok;
}

kimi_mxfp4_gpu_chain * chain_create(kimi_mxfp4_gpu * c, int latent, int hidden, int max_hits, int kernel, int keep_hidden) {
    if (!c || c->busy || c->failed) return nullptr;
    if (latent < 64 || hidden < 64 || latent % 64 || hidden % 64 || std::max(latent, hidden) > c->config.max_columns ||
            std::max(latent, hidden) > c->config.max_rows || max_hits < 1 || 2 * max_hits > c->config.max_jobs ||
            kernel < KIMI_MXFP4_CHAIN_PARITY_Q8 || kernel > KIMI_MXFP4_CHAIN_ROWS_FLOAT) {
        c->invalid("unsupported MXFP4 chain geometry, hit capacity or kernel");
        return nullptr;
    }
    device_scope device(c->config.device);
    if (!c->check(device.status, "chain device")) return nullptr;
    auto * h = new (std::nothrow) kimi_mxfp4_gpu_chain;
    if (!h) { c->invalid("MXFP4 chain host allocation"); return nullptr; }
    h->base = c; h->latent = latent; h->hidden = hidden; h->max_hits = max_hits; h->kernel = kernel; h->keep_hidden = keep_hidden != 0;
    const size_t jobs_bytes = size_t(3 * max_hits) * sizeof(device_job);
    h->input_offset = (jobs_bytes + 7) & ~size_t(7);
    h->upload_bytes = h->input_offset + size_t(latent / QK8_0) * sizeof(block_q8_0);
    h->hidden_float_bytes = size_t(max_hits) * size_t(hidden) * sizeof(float);
    h->hidden_q8_bytes = kernel == KIMI_MXFP4_CHAIN_ROWS_FLOAT ? 0 : size_t(max_hits) * size_t(hidden / QK8_0) * sizeof(block_q8_0);
    h->kept_bytes = h->keep_hidden ? (h->hidden_q8_bytes ? h->hidden_q8_bytes : h->hidden_float_bytes) : 0;
    if (!c->check(cudaMallocHost(&h->host_upload, h->upload_bytes), "chain host upload") ||
            !c->check(cudaMalloc(&h->device_upload, h->upload_bytes), "chain device upload") ||
            !c->check(cudaMalloc(&h->hidden_float, h->hidden_float_bytes), "chain hidden rows") ||
            (h->hidden_q8_bytes && !c->check(cudaMalloc(&h->hidden_q8, h->hidden_q8_bytes), "chain hidden Q8_0")) ||
            (h->kept_bytes && !c->check(cudaMallocHost(&h->host_hidden, h->kept_bytes), "chain kept hidden"))) {
        chain_destroy(h);
        return nullptr;
    }
    return h;
}

int chain_submit(kimi_mxfp4_gpu_chain * h, const kimi_mxfp4_chain_expert * experts, int hits,
                 const void * q8, size_t bytes, float limit) {
    if (!h) return 0;
    const auto c = h->base;
    if (c->failed) return 0;
    if (c->busy) return c->invalid("consume the previous event before submitting");
    if (!experts || !q8 || hits < 1 || hits > h->max_hits || bytes != size_t(h->latent / QK8_0) * sizeof(block_q8_0) ||
            !std::isfinite(limit) || !(limit > 1e-6f))
        return c->invalid("invalid MXFP4 chain hits, input or SwiGLU limit");
    for (int i = 0; i < hits; ++i) {
        if (!experts[i].up || !experts[i].gate || !experts[i].down) return c->invalid("null MXFP4 chain weights");
    }
    device_scope device(c->config.device);
    if (!c->check(device.status, "chain submit device")) return 0;
    // The tier's slots share one arena; a host-memory job's rows are host memory the device reads in place.
    for (const void * p : {experts[0].up, experts[0].gate, experts[0].down}) {
        cudaPointerAttributes attr{};
        if (!c->check(cudaPointerGetAttributes(&attr, p), "chain weight pointer")) return 0;
        if (!kimi_mxfp4_weights_readable(attr, c->config.device))
            return c->invalid("chain weights must reside on the handle device or in host memory it reads");
    }
    const bool float_hidden = h->kernel == KIMI_MXFP4_CHAIN_ROWS_FLOAT;
    const size_t up_stride = size_t(h->latent / QK_MXFP4) * sizeof(block_mxfp4);
    const size_t down_stride = size_t(h->hidden / QK_MXFP4) * sizeof(block_mxfp4);
    const size_t hidden_q8_row = size_t(h->hidden / QK8_0) * sizeof(block_q8_0);
    auto * jobs = reinterpret_cast<device_job *>(h->host_upload);
    for (int i = 0; i < hits; ++i) {
        jobs[2 * i] = {static_cast<const char *>(experts[i].up), up_stride, 0, size_t(2 * i) * size_t(h->hidden), h->latent, h->hidden};
        jobs[2 * i + 1] = {static_cast<const char *>(experts[i].gate), up_stride, 0, size_t(2 * i + 1) * size_t(h->hidden), h->latent, h->hidden};
        const size_t source = float_hidden ? size_t(i) * size_t(h->hidden) : size_t(i) * hidden_q8_row;
        jobs[2 * h->max_hits + i] = {static_cast<const char *>(experts[i].down), down_stride, source, size_t(i) * size_t(h->latent), h->hidden, h->latent};
        c->h_jobs[i] = jobs[2 * h->max_hits + i]; // result(i) reads rows and output_offset
    }
    std::memcpy(h->host_upload + h->input_offset, q8, bytes);
    c->njobs = hits;
    const auto * device_jobs = reinterpret_cast<const device_job *>(h->device_upload);
    const auto * device_down_jobs = device_jobs + 2 * h->max_hits;
    const char * device_input = h->device_upload + h->input_offset;
    c->nmarks = 0;
    c->mark(0);
    const auto h2d_start = c->timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (!c->check(cudaMemcpyAsync(h->device_upload, h->host_upload, h->upload_bytes, cudaMemcpyHostToDevice, c->stream), "chain H2D")) return 0;
    if (c->timing) c->h2d_call_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - h2d_start).count();
    c->mark(1);
    if (h->kernel == KIMI_MXFP4_CHAIN_PARITY_Q8) {
        mxfp4_avx2<<<dim3((h->hidden + 15) / 16, 2 * hits), 128, 0, c->stream>>>(device_jobs, device_input, c->d_output);
    } else {
        mxfp4_rows<<<dim3((h->hidden + chain_rows_per_block - 1) / chain_rows_per_block, 2 * hits), chain_rows_per_block, 0, c->stream>>>(
            device_jobs, device_input, c->d_output);
    }
    if (!c->check(cudaGetLastError(), "chain up/gate")) return 0;
    c->mark(2);
    chain_swiglu<<<(hits * h->hidden + 127) / 128, 128, 0, c->stream>>>(c->d_output, h->hidden_float, hits, h->hidden, limit);
    if (!c->check(cudaGetLastError(), "chain SwiGLU")) return 0;
    c->mark(3);
    if (!float_hidden) {
        chain_q8_0<<<hits * (h->hidden / QK8_0), QK8_0, 0, c->stream>>>(h->hidden_float, reinterpret_cast<block_q8_0 *>(h->hidden_q8));
        if (!c->check(cudaGetLastError(), "chain hidden Q8_0")) return 0;
        c->mark(4);
    }
    if (h->kernel == KIMI_MXFP4_CHAIN_PARITY_Q8) {
        mxfp4_avx2<<<dim3((h->latent + 15) / 16, hits), 128, 0, c->stream>>>(device_down_jobs, h->hidden_q8, c->d_output);
    } else if (!float_hidden) {
        mxfp4_rows<<<dim3((h->latent + chain_rows_per_block - 1) / chain_rows_per_block, hits), chain_rows_per_block, 0, c->stream>>>(
            device_down_jobs, h->hidden_q8, c->d_output);
    } else {
        mxfp4_float_rows<<<dim3((h->latent + chain_rows_per_block - 1) / chain_rows_per_block, hits), chain_rows_per_block, 0, c->stream>>>(
            device_down_jobs, h->hidden_float, c->d_output);
    }
    if (!c->check(cudaGetLastError(), "chain down")) return 0;
    c->mark(5);
    if (!c->check(cudaMemcpyAsync(c->h_output, c->d_output, size_t(hits) * size_t(h->latent) * sizeof(float), cudaMemcpyDeviceToHost, c->stream), "chain D2H"))
        return 0;
    if (h->keep_hidden) {
        const size_t kept = size_t(hits) * (float_hidden ? size_t(h->hidden) * sizeof(float) : hidden_q8_row);
        const void * source = float_hidden ? static_cast<void *>(h->hidden_float) : static_cast<void *>(h->hidden_q8);
        if (!c->check(cudaMemcpyAsync(h->host_hidden, source, kept, cudaMemcpyDeviceToHost, c->stream), "chain hidden D2H")) return 0;
    }
    c->mark(6);
    if (!c->check(cudaEventRecord(c->event, c->stream), "chain completion")) return 0;
    h->hits = hits;
    c->busy = true;
    return 1;
}

const void * chain_hidden(kimi_mxfp4_gpu_chain * h, size_t * bytes) {
    if (!h || !h->keep_hidden || !h->hits || query(h->base) != 1) return nullptr;
    const size_t row = h->hidden_q8_bytes ? size_t(h->hidden / QK8_0) * sizeof(block_q8_0) : size_t(h->hidden) * sizeof(float);
    if (bytes) *bytes = size_t(h->hits) * row;
    return h->host_hidden;
}
size_t chain_host_bytes(kimi_mxfp4_gpu_chain * h) { return h ? h->upload_bytes + h->kept_bytes : 0; }
size_t chain_device_bytes(kimi_mxfp4_gpu_chain * h) { return h ? h->upload_bytes + h->hidden_float_bytes + h->hidden_q8_bytes : 0; }

int chain_set_timing(kimi_mxfp4_gpu * c, int enabled) {
    if (!c || c->busy || c->failed) return 0;
    if (enabled) {
        device_scope device(c->config.device);
        if (!c->check(device.status, "timing device")) return 0;
        for (auto & m : c->marks) {
            if (!m && !c->check(cudaEventCreate(&m), "timing event")) return 0;
        }
    }
    c->timing = enabled != 0;
    c->nmarks = 0;
    return 1;
}
int chain_set_timing_stage(kimi_mxfp4_gpu * c, int stage) {
    if (!c || (stage != 1 && stage != 2)) return 0;
    c->timing_stage = stage;
    return 1;
}
int chain_timing(kimi_mxfp4_gpu * c, kimi_mxfp4_gpu_timing * out) {
    if (!c || !out) return 0;
    *out = {c->timed_submits, c->stage_ms[1], c->stage_ms[2], c->stage_ms[3], c->stage_ms[4], c->stage_ms[5], c->stage_ms[6], c->h2d_call_us};
    return 1;
}

const kimi_mxfp4_gpu_chain_api_v1 chain_api = {1, sizeof(kimi_mxfp4_gpu_chain_api_v1), chain_create, chain_submit, chain_hidden,
    chain_destroy, chain_host_bytes, chain_device_bytes, chain_set_timing, chain_set_timing_stage, chain_timing};
}

extern "C" KIMI_MXFP4_EXPORT const kimi_mxfp4_gpu_chain_api_v1 * kimi_mxfp4_gpu_get_chain_api_v1() { return &chain_api; }
