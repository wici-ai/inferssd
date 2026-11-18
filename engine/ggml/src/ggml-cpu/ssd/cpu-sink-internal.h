#pragma once

#include "cpu-sink.h"
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
bool ssd_cpu_sink_stream_hybrid(struct ssd_cpu_sink * sink,
        const struct ggml_tensor * up, const struct ggml_tensor * gate,
        const struct ggml_tensor * down, const int32_t * ids, int nexperts,
        struct ssd_cpu_sink_prefill_batch * batch,
        ssd_cpu_sink_consumer_fn consumer, void * userdata,
        struct ssd_cpu_sink_stats * stats, char * error, size_t error_size);

// Borrowed immutable selection; the prefill owner never outlives its sink.
const ssd::telemetry::debug_channels & ssd_cpu_sink_debug(const ssd_cpu_sink * sink) noexcept;
