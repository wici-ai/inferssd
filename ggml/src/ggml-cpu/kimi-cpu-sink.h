#pragma once

#include "ggml-cpu.h"
#include "kimi-cpu-cache-policy.h"

#ifdef __cplusplus
extern "C" {
#endif

struct kimi_cpu_sink;

// Borrowed descriptors must remain open until run returns. No tensor payload is
// accessed: resolve the packed expert by its tensor name and expert index.
struct kimi_cpu_sink_source {
    int fd_direct;
    uint64_t offset;
    size_t nbytes;             // physical aligned read capacity
    size_t data_offset;        // logical payload starts this far into the read
    size_t payload_nbytes;     // zero retains the legacy whole-read payload
};

// The model-side native-range resolver checks this before using the callback ABI.
GGML_BACKEND_API size_t kimi_cpu_sink_source_size(void);

typedef bool (*kimi_cpu_sink_resolver)(
        void * userdata, const char * tensor_name, int32_t expert,
        struct kimi_cpu_sink_source * source);

struct kimi_cpu_sink_stats {
    uint64_t experts;
    uint64_t read_requests;
    uint64_t read_bytes;
    uint64_t read_errors;
    uint64_t fallback_count; // always zero; errors never use tensor->data
    uint64_t ring_bytes;
    double total_us;
    double io_us;           // I/O job wall time, overlaps compute_us
    double compute_us;      // conversion + CPU phases, excluding I/O waits
    double wait_io_us;
    double input_quantize_us;
    double hidden_quantize_us;
    double up_gate_us;
    double down_reduce_us;
    uint64_t demand_bytes;       // all live slots, including cache hits
    uint64_t cache_hits;         // only entries resident before this token
    uint64_t cache_hit_bytes;
    uint64_t cache_misses;
    uint64_t cache_admitted;
    uint64_t cache_admitted_bytes;
    double cache_copy_us;        // zero: admissions read directly into cache slots
    uint64_t cache_fill_read_requests; // submitted SSD reads targeting cache payload
    uint64_t cache_fill_read_bytes;    // positive CQE bytes, including a possible short read
    uint64_t ring_read_requests;
    uint64_t ring_read_bytes;
    uint64_t fixed_read_requests;      // READ_FIXED SQEs; only ring targets are eligible
    uint64_t read_completions;         // consumed CQEs, including error-path drain
    uint64_t pending_reads;            // must be zero when run returns, including errors
    uint64_t cache_new_slot_bytes;     // previously untouched slots reserved as read targets
    // PROFILE-only samples. The first three are thread CPU time, not wall time.
    // compute_cpu_us includes all worker kernels and decode quantization; caller
    // overlaps worker 0 and quantization, so those counters must not be added.
    double compute_cpu_us;
    double caller_cpu_us;
    double io_cpu_us;
    double io_wait_us;        // io_uring GETEVENTS wall time, overlaps compute
    double io_credit_wait_us; // I/O thread waiting for a compute-owned ring slot, nothing in flight (always set)
    double other_us;          // total_us - compute_us - wait_io_us (caller wall)
    uint64_t ready_scratch_bytes; // retained vector capacities, zero when disabled
    uint64_t ready_out_of_order_experts; // executed at a different ordinal
    double ordered_reduce_us; // ready-first final slot-order sum; inside compute_us
};

struct kimi_cpu_sink_cache_stats {
    uint64_t budget_bytes;       // hard payload mapping limit; metadata/ring are separate
    uint64_t capacity_bytes;
    uint64_t mapped_bytes;
    uint64_t committed_bytes;    // upper bound: resident slots; MADV_DONTNEED removes them
    uint64_t enabled_bytes;      // slots eligible for hits/admission in the current phase
    uint64_t valid_bytes;
    uint64_t peak_bytes;
    uint64_t hits;
    uint64_t misses;
    uint64_t admitted_bytes;
    uint64_t evictions;
    uint64_t admission_skips;
    uint64_t clears;
    uint64_t pressure_stops;
    double copy_us;
    uint64_t page_table_bytes;   // mapping-lifetime upper bound for touched slots; not measured RSS
    uint64_t last_guard_extra_bytes; // most recent incremental cgroup check, including allocation overhead
    uint64_t peak_guard_extra_bytes;
    uint64_t shrink_ops;             // successful phase/pressure shrink operations
    uint64_t phase_grows;            // phase-elastic decode logical grows (no prefault)
    uint64_t released_bytes;         // cumulative resident payload released
    uint64_t refaulted_bytes;        // cumulative released payload reserved again
    uint64_t guard_reclaims;         // shrink operations initiated by the guard
    uint64_t shrink_us;              // cumulative synchronous shrink wall time
    uint64_t prefill_hits;            // read-only prefill borrows; excluded from decode heat
    uint64_t prefill_misses;
    uint64_t prefill_hit_bytes;
    uint64_t prefill_leases;          // successfully acquired read-only leases
};

// Linux io_uring and O_DIRECT are required. total_threads includes the calling
// compute thread and one I/O thread; all other threads sleep between phases.
// A context serializes run calls. Suggested values: total_threads=16,
// ring_experts=8. Optional GGML_KIMI_CPU_SINK_ environment settings at create:
// READY_FIRST=1 computes available experts first, then adds in original order.
// EARLY_UP_GATE=1 may start a miss after its up and gate reads complete, then
// waits for down before that projection. It requires EXPERT_JOB=0 and cannot
// be combined with the cache-location provider; the native GPU tier is safe.
// ROLE_MAJOR=1 queues every up/gate pair in an I/O batch before its down reads.
// It requires EARLY_UP_GATE=1; without early compute the order is detrimental.
// NONBLOCKING_DOWN_SCHED=1 splits each CPU expert into up/gate+quantization and
// down work, always running an available first half instead of waiting on one
// late down read. Per-expert quantized hidden rows preserve that dependency.
// It requires READY_FIRST=1, EARLY_UP_GATE=1, ROLE_MAJOR=1 and EXPERT_JOB=0.
// ROW_WEIGHTS is a positive finite comma-separated weight per compute participant
// (total_threads - 1, caller first). It affects decode row partitioning only.
// DYNAMIC_ROWS=1 instead lets every compute participant claim aligned row blocks
// from a shared cursor; it is mutually exclusive with ROW_WEIGHTS and does not
// alter the ith/nth prefill callback ABI. DYNAMIC_CHUNK_ROWS selects a multiple
// of 16 rows in [16,16384] (default 64). Dynamic scheduling defaults off.
// COMPUTE_CPUS is one logical CPU per compute participant; IO_CPUS is the allowed
// CPU list for the sole I/O thread. Both use comma-separated IDs in the creator's
// allowed mask. Invalid settings fail closed. Empty lists disable these options.
// Caller affinity is scoped to run/stream and restored on return; persistent
// workers retain their explicit affinity. Defaults preserve OS scheduling.
GGML_BACKEND_API struct kimi_cpu_sink * kimi_cpu_sink_create(
        int total_threads, int ring_experts, kimi_cpu_sink_resolver resolver,
        void * userdata, char * error, size_t error_size);
GGML_BACKEND_API void kimi_cpu_sink_destroy(struct kimi_cpu_sink * sink);
// Next-layer expert prefetch (GGML_KIMI_CPU_SINK_PREFETCH=k): the experts that `layer` is predicted to select,
// best first, given before the decode run of the layer before it. Ignored when prefetch is off.
GGML_BACKEND_API void kimi_cpu_sink_prefetch_hint(struct kimi_cpu_sink * sink, int32_t layer, const int32_t * experts, int n);

// Configure while no run is in progress. Layer quotas are assigned in priority
// order, up to max_experts_per_layer each; a final layer may receive a partial
// quota. Layer IDs refer to canonical blk.N.ffn_{up,gate,down}_exps.weight names.
// Payload uses ordinary anonymous mmap, never pinned/registered. Configuration
// does not touch payload pages. budget=0 disables the cache. Reconfiguration
// releases the old mapping first. max_experts_per_layer must be 1..896.
// Admission requires readable finite cgroup-v2 memory limits and, by default,
// 2 GiB spare headroom at every finite ancestor. Checks charge only this batch's newly
// resident slots, plus conservative page-table bounds and allocator allowances;
// the full virtual payload budget is never prefaulted or charged as resident.
// Without partial reclaim, a pressure stop releases the mapping and pauses
// until decode=false. CACHE_RESERVE_BYTES accepts uint64 byte counts >=512 MiB; default 2 GiB.
// A reserve larger than the finite cgroup headroom disables optional cache admission.
// Caller must disable swapping (MemorySwapMax=0) and keep cgroup membership
// fixed: first-touch accounting assumes resident slots cannot be swapped out
// and faulted back in. Concurrent non-cache growth can consume the reserve.
// At create time, CACHE_POLICY (GGML_KIMI_CPU_SINK_ prefix) selects window
// (default), lru, lfu, decay (LFU counts halved every 32 decode tokens), or
// count_decay2 (unweighted selection frequency, half-life two decode tokens).
// CACHE_KEEP_ACROSS_REQUESTS=1 retains persistent-policy payload at prefill
// boundaries when the selected reserve permits; default 0 releases it.
// CACHE_RECLAIMABLE_FILE=1 subtracts each bounded cgroup's active_file and
// inactive_file bytes from memory.current in that guard; default 0. Unreadable
// controller/stat files still fail closed.
// CACHE_PARTIAL_RECLAIM=1 requires persistent policy + uniform allocation +
// CACHE_KEEP_ACROSS_REQUESTS=1. A known guard shortage disables low-score
// slots within every layer and releases their resident pages with
// MADV_DONTNEED. It leaves the mapping and policy heat intact; it does not grow
// those slots back. An unreadable guard still rejects admission fail-closed.
// CACHE_ELASTIC=1 has the same configuration requirements and additionally
// requires CACHE_PREFILL_SLOTS. At prefill it keeps exactly that many enabled
// slots, chosen per layer by policy score. The first following decode logically
// enables every slot without prefaulting. Partial reclaim and phase elasticity
// are independent opt-ins and are disabled by default.
// Prefill never reads/adopts these entries. Frequency age counts decode tokens
// across retained requests; prefill tokens do not train or age the policy.
// CACHE_ALLOCATION selects priority (default), uniform, or global. Global is
// a shared pool for persistent policies; max_experts_per_layer only bounds its
// total capacity. Uniform divides slots evenly, with remainders in priority
// order. Every policy protects the complete current batch before reserving
// zero-copy read targets. LFU counts include misses; lower-frequency misses
// bypass a full cache, and equal-frequency victims are ordered by recency.
GGML_BACKEND_API bool kimi_cpu_sink_cache_configure(
        struct kimi_cpu_sink * sink, uint64_t budget_bytes,
        const int32_t * layer_priority, int n_layers, int max_experts_per_layer,
        char * error, size_t error_size);

// Call once before each logical token, and with decode=false before prefill or
// an unsupported batch. False disables cache use and by default unmaps pages;
// optional retention keeps weights only while headroom permits. A changed epoch
// (model/context identity), or nonconsecutive token without an intervening false,
// clears the mapping. Callers
// must serialize begin/configure/run as one sequence. Packs must be immutable
// during an epoch; replacing weights requires a new epoch even at the same fd.
GGML_BACKEND_API bool kimi_cpu_sink_cache_begin(
        struct kimi_cpu_sink * sink, uint64_t epoch, uint64_t token, bool decode,
        char * error, size_t error_size);
// Completes the phase transition after the scheduler reserve. decode=false
// only disarms the prefill OOM reclaimer; decode=true also grows the optional
// global VRAM tier. No-op when the elastic extension is unavailable or off.
GGML_BACKEND_API bool kimi_cpu_sink_vram_phase_complete(
        struct kimi_cpu_sink * sink, bool decode, char * error, size_t error_size);
// Reserve the GPU batch accumulator for this actual ubatch before streamed I/O begins.
GGML_BACKEND_API bool kimi_cpu_sink_vram_reserve_accumulator(
        struct kimi_cpu_sink * sink, int tokens, char * error, size_t error_size);
// At load, after the context reserved its worst-case graph: on unified memory the GPU tier takes its storage for
// these routed experts (one layer's tensors give the format) and grows into what the host-memory ledger leaves,
// after the RAM cache's demand; elsewhere (with a GPU tier or without) a RAM cache kept across requests that can give
// room back (CACHE_PARTIAL_RECLAIM or CACHE_ELASTIC) takes its geometry from the tensors' strides, its arena and the
// pages of the slots the ledger admits now. A cache that cannot give room back takes nothing here: it is sized at
// the first decode token, after the first prefill's ledger requests, which an arena made now would refuse.
// experts_used: the model's routed experts a token.
GGML_BACKEND_API bool kimi_cpu_sink_vram_reserve(struct kimi_cpu_sink * sink, const struct ggml_tensor * up,
        const struct ggml_tensor * gate, const struct ggml_tensor * down, int experts_used, char * error,
        size_t error_size);
GGML_BACKEND_API void kimi_cpu_sink_cache_get_stats(
        struct kimi_cpu_sink * sink, struct kimi_cpu_sink_cache_stats * stats);
// Room in host memory for other memory about to be allocated, decided by the host-memory ledger
// (kimi-host-ledger.h): `host_bytes` of host memory (host compute buffers, checkpoints, prompt-cache blobs) and
// `device_bytes` of GPU memory (device compute buffers), which count only where the GPU allocates out of the same
// DRAM. The tightest finite cgroup limit and MemAvailable less the floor must both still hold. On a unified-memory
// GPU the GPU tier gives blocks back first, then RAM cache slots go. Returns false when either would still be
// exceeded. With a null sink (before the first graph creates one) the ledger alone decides.
GGML_BACKEND_API bool kimi_cpu_sink_cache_make_room(struct kimi_cpu_sink * sink, uint64_t host_bytes, uint64_t device_bytes);
// Explicit reset/errors always release, even when cross-request retention is on.
GGML_BACKEND_API void kimi_cpu_sink_cache_reset(struct kimi_cpu_sink * sink);

// Single token. up/gate=[latent,hidden,experts], down=[hidden,latent,experts].
// Tensors must be contiguous. Kernels come from each tensor's CPU type traits:
// up and gate share one type (e.g. Q3_K with a Q8_K input, MXFP4 with a Q8_0
// input); down may use another. Column counts must be multiples of the weight
// and input block sizes. ids<0 are skipped; each live weight must be finite and
// nonnegative. Caller supplies the final normalized weights. Each expert's down
// output is multiplied by its weight, then added in slot order. On failure
// output must be discarded. This is kimi_cpu_sink_run_activation with SITU.
GGML_BACKEND_API bool kimi_cpu_sink_run(
        struct kimi_cpu_sink * sink,
        const struct ggml_tensor * up, const struct ggml_tensor * gate,
        const struct ggml_tensor * down, const float * input,
        const int32_t * ids, const float * weights, int nslots,
        float situ_beta, float situ_linear_beta, float * output,
        struct kimi_cpu_sink_stats * stats, char * error, size_t error_size);

// Routed-expert activation between the up/gate and down projections.
// SITU (Kimi-K3): beta*tanh(g/beta)*sigmoid(g) * lb*tanh(u/lb), where lb <= 0
// keeps u linear; beta must be finite and positive and lb finite.
// SWIGLU_CLAMP (DeepSeek-V4.1): silu(min(g, L)) * clamp(u, -L, L), computed as
// the scalar GGML_GLU_OP_SWIGLU_CLAMP CPU kernel; L must be > 1e-6; +infinity is plain SwiGLU.
enum kimi_cpu_sink_activation_kind {
    KIMI_CPU_SINK_ACTIVATION_SITU         = 0,
    KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP = 1,
};

struct kimi_cpu_sink_activation {
    int32_t kind; // enum kimi_cpu_sink_activation_kind
    float   situ_beta;
    float   situ_linear_beta;
    float   swiglu_limit; // +infinity means unclamped SwiGLU
    const float * post_exp_norm; // optional latent-length weight, borrowed for run
    float post_exp_norm_eps;
};

// kimi_cpu_sink_run with an explicit activation. A configured GPU tier serves
// Q3_K SiTU experts through its original ABI and other expert formats only when
// it exports kimi_vram_format_get_api; otherwise the call fails closed.
GGML_BACKEND_API bool kimi_cpu_sink_run_activation(
        struct kimi_cpu_sink * sink,
        const struct ggml_tensor * up, const struct ggml_tensor * gate,
        const struct ggml_tensor * down, const float * input,
        const int32_t * ids, const float * weights, int nslots,
        const struct kimi_cpu_sink_activation * activation, float * output,
        struct kimi_cpu_sink_stats * stats, char * error, size_t error_size);

// Optional shared-location provider. nullptr detaches at a serialized boundary.
// Default absent; the placement owner installs its adapter once, not per token.
struct kimi_cache_location_api;
GGML_BACKEND_API bool kimi_cpu_sink_set_cache_locations(
        struct kimi_cpu_sink *, const struct kimi_cache_location_api *, void *, char *, size_t);

// Separate ABI: the original stats and run entry remain unchanged.
struct kimi_cache_compute_stats {
    struct kimi_cache_compute_plan plan;
    uint32_t computed_mask;
    uint32_t released_mask;
    uint64_t route_visible_us;
    uint64_t prepared_us;
    uint64_t finished_us;
    uint64_t baseline_miss_bytes;
    uint64_t submitted_bytes;
    uint64_t unused_read_bytes;
    // Native slot: queue, submit begin/end, up/gate/down CQE, CPU begin/end.
    // For registered VRAM jobs, the final pair is a batch submit-to-finish
    // host envelope, not per-expert CUDA kernel timing (see trace vram_mask).
    uint64_t events[16][8];
    uint64_t completed_bytes[16];
    float final_weights[16];
};
GGML_BACKEND_API bool kimi_cpu_sink_run_cache_compute(
        struct kimi_cpu_sink * sink,
        const struct ggml_tensor * up, const struct ggml_tensor * gate,
        const struct ggml_tensor * down, const float * input,
        const int32_t * ids, const float * weights, int nslots,
        const int32_t * native_ids, const float * native_weights, int native_count,
        float routed_scale, bool guarded, const struct kimi_cache_compute_config * config,
        float situ_beta, float situ_linear_beta, float * output,
        struct kimi_cpu_sink_stats * stats, struct kimi_cache_compute_stats * decision,
        char * error, size_t error_size);

#ifdef __cplusplus
}
#endif
