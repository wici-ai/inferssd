#pragma once

#include "cpu-sink.h"
#include "ggml-cpu.h"
#include "ggml-ssd/store/expert-layout.h"
#include <cmath>
#include <cstdint>
#include <functional>
#include "ggml-ssd/telemetry/telemetry.h"

// Matches mul(weight) followed by ggml_rms_norm, which squares in float and sums in double, then route scaling.
inline void ssd_cpu_sink_normalize_expert(float * values, int rows, const float * norm, float eps) {
    double sum = 0.0;
    for (int row = 0; row < rows; ++row) {
        values[row] *= norm[row];
        sum += double(values[row] * values[row]);
    }
    const float scale = 1.0f / std::sqrt(float(sum / rows) + eps);
    for (int row = 0; row < rows; ++row) { values[row] *= scale; }
}


// Private executor hooks. The stream serializes all accesses to the shared
// decode ring and compute phases. Call parallel only from the stream consumer,
// and parallel_last only from the caller of a stream that returned, as the
// open call's last phase; callbacks must not recursively enter either executor
// or allocate/throw.
// Streaming is a prefill operation: it disables decode cache (normally releases
// it; optional cross-request retention checks the same memory guard) and
// always uses ring targets. Cache admission requires a subsequent decode begin.
// The CPU's kernels for one format of the sink's expert layout, chosen when the sink is created: each role's vec_dot
// against a row of the role's input quantized to its vec_dot_type, the latent's for up and gate (one quantization of
// the input serves both, so their vec_dot_types match) and the hidden's for down.
struct ssd_cpu_expert_kernels {
    ggml_vec_dot_t dot[SSD_EXPERT_ROLES];
    ggml_type input_type, hidden_type;                // the vec_dot_types of up (and gate) and of down
    ggml_from_float_t quantize_input, quantize_hidden;
    size_t row_bytes[SSD_EXPERT_ROLES];               // a weight row of each role
    size_t input_row_bytes, hidden_row_bytes;         // a quantized latent row, a quantized hidden row
    bool mxfp4;                                       // every role is MXFP4
};
// The sink's expert layout, and the kernels of routed layer `layer`'s format (null for any other layer).
const ssd_expert_layout & ssd_cpu_sink_layout(const struct ssd_cpu_sink * sink) noexcept;
const ssd_cpu_expert_kernels * ssd_cpu_sink_kernels(const struct ssd_cpu_sink * sink, int32_t layer) noexcept;

using ssd_cpu_sink_parallel_fn = void (*)(void * userdata, int ith, int nth);
using ssd_cpu_sink_consumer_fn = void (*)(void * userdata, size_t expert,
        const void * up, const void * gate, const void * down);

struct ssd_cpu_sink_prefill_batch {
    const float * f32_input;
    size_t f32_stride;
    const int32_t * route_request_index;
    const float * weights;
    int nslots;
    int ntokens;
    struct ssd_cpu_sink_activation activation;
    float * output;
    uint64_t gpu_experts;
    uint64_t gpu_routes;
    uint64_t gpu_transient_experts;
    uint64_t gpu_transient_routes;
    uint64_t gpu_transient_upload_bytes;
    double gpu_transient_upload_us;
    uint64_t gpu_ssd_transient_candidates;
    uint64_t gpu_ssd_transient_experts;
    uint64_t gpu_ssd_transient_routes;
    uint64_t gpu_ssd_transient_upload_bytes;
    double gpu_ssd_transient_upload_us;
    double gpu_worker_us;
    double gpu_expert_us;    // the GPU's time in its expert chains
    double gpu_read_wait_us; // the GPU worker's wait for streamed SSD reads
    uint64_t ram_experts;
    uint64_t ram_routes;
    uint64_t ram_bytes;
    uint64_t ssd_experts;
    uint64_t ssd_routes;
    uint64_t ssd_bytes;
};

void ssd_cpu_sink_parallel(struct ssd_cpu_sink * sink,
        ssd_cpu_sink_parallel_fn function, void * userdata);
// Runs grow, which allocates and first touches `bytes` of host memory outside the caches, under a ticket of the sink's
// host-memory context: the ledger makes room for them (the RAM cache gives slots back) and refuses only what the
// envelope cannot hold, as the sink's own scratch is priced. Throws the refusal; call outside a run.
void ssd_cpu_sink_host_grow(struct ssd_cpu_sink * sink, uint64_t bytes, const std::function<void()> & grow);
void ssd_cpu_sink_parallel_last(struct ssd_cpu_sink * sink,
        ssd_cpu_sink_parallel_fn function, void * userdata);
bool ssd_cpu_sink_stream_hybrid(struct ssd_cpu_sink * sink, int32_t layer, const int32_t * ids, int nexperts,
        struct ssd_cpu_sink_prefill_batch * batch,
        ssd_cpu_sink_consumer_fn consumer, void * userdata,
        struct ssd_cpu_sink_stats * stats, char * error, size_t error_size);

// Borrowed immutable selection; the prefill owner never outlives its sink.
const ssd::telemetry::debug_channels & ssd_cpu_sink_debug(const ssd_cpu_sink * sink) noexcept;

// The context's model-scoped policy; first prefill preparation emits its activation receipt.
bool ssd_cpu_sink_prefill_union_budget(ssd_cpu_sink * sink);
void ssd_cpu_sink_prefill_ready(ssd_cpu_sink * sink, int32_t layer,
        const int32_t * ids, int count, uint8_t * ready);
