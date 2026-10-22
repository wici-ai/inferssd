#pragma once
#include <stddef.h>
#include <stdint.h>
#include <sys/stat.h>

// Private optional transport ABI. The CPU backend remains CUDA independent.
// Every function except destroy returns 0 on success. Errors must drain work
// before payload/input/output ownership is returned to the caller.
struct kimi_vram_source {
    uint64_t device, inode, size, offset, bytes;
    int64_t mtime_s, mtime_ns, ctime_s, ctime_ns;
};
struct kimi_vram_request {
    struct kimi_vram_source role[3]; // up, gate, down, same packed-file identity
    int32_t layer, expert;
    float weight;
};
// A role's payload as the RAM cache and the VRAM tier know it: the file behind the expert-pack resolver's
// descriptor (st, from fstat), the role's logical offset in it (where the resolver says its bytes start, before a
// read aligns it) and its logical bytes. The sink keys what it reads by it and a prefill seed what it copies, so
// that decode finds the seed: both build it here.
static inline struct kimi_vram_source kimi_vram_source_identity(const struct stat * st, uint64_t offset, uint64_t bytes) {
    struct kimi_vram_source s = { (uint64_t) st->st_dev, (uint64_t) st->st_ino, (uint64_t) st->st_size, offset, bytes,
        (int64_t) st->st_mtim.tv_sec, (int64_t) st->st_mtim.tv_nsec, (int64_t) st->st_ctim.tv_sec, (int64_t) st->st_ctim.tv_nsec };
    return s;
}
struct kimi_vram_tier_api {
    uint32_t version; // 2
    uint32_t size;
    void * (*create)(uint64_t budget_bytes, char * error, size_t error_size);
    void (*destroy)(void * context);
    int (*begin)(void * context, uint64_t epoch, uint64_t token, int decode,
                 char * error, size_t error_size);
    // prepare pins hits and selects this layer's admission candidates. slots
    // receives -1 for CPU work, otherwise an opaque pinned GPU cache slot.
    int (*prepare)(void * context, const struct kimi_vram_request * requests,
                   int count, int latent, int hidden, int * slots,
                   char * error, size_t error_size);
    // launch copies Q8_K input before returning. A private worker drives the
    // two GPU stages and CPU SiTU bridge while CPU experts and SSD proceed.
    int (*launch)(void * context, const void * input_q8k, size_t input_bytes,
                  float beta, float linear_beta, char * error, size_t error_size);
    // A RAM cache source may be borrowed until finish/cancel when borrow=1;
    // the caller pins its slot for this entire batch. Ring sources use borrow=0
    // and must be completely copied before return so I/O may reuse the ring.
    int (*admit)(void * context, const struct kimi_vram_request * request,
                 const void * up, const void * gate, const void * down,
                 int borrow_until_finish, char * error, size_t error_size);
    // Only original GPU-hit rows are written; CPU rows remain untouched.
    // Contributions include the separately rounded routing-weight multiply.
    // stride is floats. finish waits for owned GPU work and unpins all hits.
    int (*finish)(void * context, float * contributions, size_t stride,
                  char * error, size_t error_size);
    int (*cancel)(void * context, char * error, size_t error_size);
};
// Optional extension, separate from transport ABI 2. acquire may only follow a
// successful finish. It pins each returned ready copy while RAM decides whether
// to drop its duplicate; pending uploads never count as ready. The caller owns
// all metadata under the sink run mutex and must release its CPU/I/O/DMA leases
// before committing a RAM drop. No begin/prepare may occur with a live lease.
// held (version 2): which of `count` experts of `layer` the tier holds now (out[i] 1 or 0): a slot has its bytes,
// or is reserved for a deferred upload of them that the layer's next prepare collects first, so that prepare would
// pin it as a hit. A seed counts from its publication: a prefill's planned seeds when the decode phase completes the
// prefill, Q2's cold-prefill admission right after its copy. No lease and any phase: the next-layer prefetch asks
// before it reads an expert into RAM, and the answer may change at the next admission, eviction or resize.
struct kimi_vram_ready_lease { uint64_t epoch, generation; };
struct kimi_vram_residency_api {
    uint32_t version, size;   // 2
    int (*acquire)(void * context, uint64_t epoch, const struct kimi_vram_request * requests,
                   int count, struct kimi_vram_ready_lease * lease, int * ready,
                   char * error, size_t error_size);
    int (*validate)(void * context, const struct kimi_vram_ready_lease * lease,
                    char * error, size_t error_size);
    int (*release)(void * context, const struct kimi_vram_ready_lease * lease,
                   char * error, size_t error_size);
    void (*held)(void * context, int32_t layer, const int32_t * experts, int count, uint8_t * out);
};
// Optional explicit-format extension, separate from transport ABI 2. The ABI 2
// prepare/launch entries remain the Kimi-K3 Q3_K/SiTU path. For any other expert
// format the CPU owner calls prepare (with the tensor geometry) instead of ABI 2
// prepare and launch (with the activation) instead of ABI 2 launch; admit,
// finish, cancel, begin and the residency ABI are shared. The format, including
// the layer list, must stay the same for the lifetime of a context.
enum kimi_vram_activation_kind {
    KIMI_VRAM_ACTIVATION_SITU         = 0, // same values as kimi_cpu_sink_activation_kind
    KIMI_VRAM_ACTIVATION_SWIGLU_CLAMP = 1,
};
struct kimi_vram_activation {
    int32_t kind;
    float situ_beta, situ_linear_beta, swiglu_limit;
};
struct kimi_vram_expert_format {
    int32_t latent, hidden;   // up/gate columns and rows
    int32_t weight_type;      // ggml_type of up and gate
    int32_t down_type;        // ggml_type of down
    int32_t n_experts;        // routed experts per layer (tensor ne[2])
    int32_t n_layers;         // entries in layers
    const int32_t * layers;   // model layer IDs, allocation priority order
};
struct kimi_vram_format_api {
    uint32_t version, size;   // 1
    int (*prepare)(void * context, const struct kimi_vram_request * requests, int count,
                   const struct kimi_vram_expert_format * format, int * slots,
                   char * error, size_t error_size);
    // input is the up weight type's CPU vec_dot input row for the original
    // explicit engines (Q8_0 for MXFP4), or an F32 row when the selected
    // format engine documents that requirement (IQ2_XXS/Q2_K).
    int (*launch)(void * context, const void * input, size_t input_bytes,
                  const struct kimi_vram_activation * activation, char * error, size_t error_size);
};
// One host-resident expert selected for transient GPU prefill compute. The
// request index addresses the distinct-expert union passed to prepare. RAM
// pointers stay leased by the cache owner. Q2 SSD pointers use dedicated
// staging through finish/cancel; MXFP4 SSD pointers use a recyclable I/O ring
// slot and the stage callback below. The tier copies into a dedicated device
// scratch slot without changing decode residency.
enum kimi_vram_prefill_transient_source {
    KIMI_VRAM_PREFILL_TRANSIENT_RAM = 0,
    KIMI_VRAM_PREFILL_TRANSIENT_SSD = 1,
};
struct kimi_vram_prefill_transient {
    int32_t request_index;
    int32_t source;
    const void * role[3];
    // Optional SSD ring handoff: phase 0 waits for the read, phase 1 returns
    // its I/O credit after the GPU upload has finished, and phase 2 probes
    // readiness without waiting. Zero for leased RAM and for the original
    // dedicated-staging path.
    int (*stage)(void * context, int32_t request_index, int phase);
    void * stage_context;
};
struct kimi_vram_prefill_stats {
    uint64_t resident_experts;
    uint64_t resident_routes;
    uint64_t transient_experts;
    uint64_t transient_routes;
    uint64_t transient_upload_bytes;
    double transient_upload_us;
    uint64_t ram_transient_experts;
    uint64_t ram_transient_routes;
    uint64_t ram_transient_upload_bytes;
    double ram_transient_upload_us;
    uint64_t ssd_transient_experts;
    uint64_t ssd_transient_routes;
    uint64_t ssd_transient_upload_bytes;
    double ssd_transient_upload_us;
    double total_worker_us;
    // Version 5, MXFP4 batch path only. The worker's time in each expert computation (its routes' gather, the
    // kernels and their synchronisation) as sums over the experts, routes = the expert's distinct tokens: count, sum
    // of routes, of routes squared, of the times, and of routes times the time; the worker's wait for streamed SSD
    // reads; the GPU output's addition into the caller's rows at finish.
    double gpu_fit_experts;
    double gpu_fit_routes;
    double gpu_fit_routes_sq;
    double gpu_fit_us;
    double gpu_fit_route_us;
    double read_wait_us;
    double output_add_us;
};
// Optional grouped-prefill extension. Residency hits are pinned read-only.
// Version 2 can additionally consume selected, already-leased RAM-resident Q2
// experts through a dedicated transient scratch slot. Version 3 can seed empty
// decode slots from selected cold-prefill SSD sources. Version 4 also accepts
// completed O_DIRECT reads from dedicated SSD staging buffers as transient Q2
// compute sources. Version 5 adds the MXFP4 worker's per-expert times, read waits and output addition to the
// stats. Cold seeds start with zero decode heat and are therefore the
// first victims unless decode reuses them; the path never evicts an existing
// decode resident.
struct kimi_vram_prefill_api {
    uint32_t version, size;   // 5
    // requests contains the distinct expert union for one layer. slots receives
    // -1 for CPU work or an opaque resident slot for GPU work. The tier must
    // already hold the same explicit format from an earlier decode phase.
    int (*prepare)(void * context, const struct kimi_vram_request * requests, int count,
                   const struct kimi_vram_expert_format * format, int * slots,
                   char * error, size_t error_size);
    // route_request_index is token-major [ntokens][nslots] and indexes requests;
    // -1 skips a route. input contains ntokens rows separated by input_stride.
    // launch copies all caller-owned inputs and route metadata before returning.
    // Returns one only when a dedicated transient scratch slot exists for this
    // context/format. It is a read-only query at a serialized phase boundary.
    int (*transient_available)(void * context);
    int (*launch)(void * context, const void * input, size_t input_stride,
                  const int32_t * route_request_index, const float * weights,
                  int nslots, int ntokens,
                  const struct kimi_vram_prefill_transient * transient, int transient_count,
                  const struct kimi_vram_activation * activation,
                  char * error, size_t error_size);
    // Adds the GPU-resident routes to token-major output rows. Existing output
    // values are preserved, so the CPU miss executor may run concurrently.
    int (*finish)(void * context, float * output, size_t output_stride,
                  struct kimi_vram_prefill_stats * stats,
                  char * error, size_t error_size);
    int (*cancel)(void * context, char * error, size_t error_size);
    // True only for a tier initialized by the current cold prefill. admit_cold
    // copies before returning, so an O_DIRECT ring source may be recycled as
    // soon as the call succeeds. *admitted is zero when no empty slot remains.
    int (*cold_admission_active)(void * context);
    int (*admit_cold)(void * context, const struct kimi_vram_request * request,
                      const void * up, const void * gate, const void * down,
                      int * admitted, char * error, size_t error_size);
};
// Optional deferred-upload extension (GGML_KIMI_CPU_SINK_VRAM_DEFERRED_UPLOAD=1 in the tier and its CPU owner,
// with ASYNC_UPLOAD=1). admit replaces ABI 2 admit for a borrowed RAM source, but the upload may outlive finish:
// when *queued is 1 the caller keeps that source unchanged and leased until collect returns the request, and the
// reserved GPU slot is never evicted meanwhile. collect runs outside prepare..finish and without a live residency
// lease. It waits for the in-flight uploads of wait_layer (>= 0) or of every layer (-1; -2 waits for none), then
// publishes completed uploads and copies their keys into completed (at most capacity; call again while
// count == capacity). aborted[i] is 1 when that upload's slot went with a block the tier gave back (an OOM or host
// reclaim, which waits for the copy first): nothing was published, but its RAM source is free again like the others.
// wait blocks, in any phase, until no upload is in flight, without publishing; call it before the RAM owner unmaps
// a source. begin for prefill, release, cancel and destroy wait as well.
struct kimi_vram_deferred_api {
    uint32_t version, size;   // 2
    int (*admit)(void * context, const struct kimi_vram_request * request, const void * up, const void * gate,
                 const void * down, int * queued, char * error, size_t error_size);
    int (*collect)(void * context, int32_t wait_layer, struct kimi_vram_request * completed, int32_t * aborted,
                   int capacity, int * count, char * error, size_t error_size);
    int (*wait)(void * context, char * error, size_t error_size);
};
// Optional lifecycle extension, separate from the compute/transport ABIs.
// begin(decode=false) performs the proactive request-boundary shrink and arms
// allocator recovery. phase_complete(false) disarms after a successful prefill
// reserve; phase_complete(true) runs only after the decode graph is reserved
// and grows best-effort into currently free device memory.
struct kimi_vram_elastic_api {
    uint32_t version, size;   // 3
    int (*phase_complete)(void * context, int decode, char * error, size_t error_size);
    // Reserve the actual grouped-prefill accumulator before starting streamed I/O.
    int (*reserve_accumulate)(void * context, int tokens, char * error, size_t error_size);
    // At load, after the context reserved its worst-case graph and the RAM cache published its demand: on unified
    // memory the tier takes its storage and grows into what the host-memory ledger leaves, so that no request waits
    // for the page population; the next decode adopts it. A discrete GPU keeps its arena where it is and returns.
    // The request gives the role bytes (one expert of a routed layer), format the geometry.
    int (*reserve)(void * context, const struct kimi_vram_request * request,
                   const struct kimi_vram_expert_format * format, char * error, size_t error_size);
};
// Optional decode extension: the GPU computes experts that sit in host memory (a RAM cache slot or a ring
// buffer). How depends on the device, and costs2 says which:
//   - a device that reads host memory in place through the host page tables (ATS: GB10, Grace Hopper; not HMM,
//     which migrates the pages) computes from the host pointers;
//   - a discrete GPU (version 2) copies each job from page-locked host memory (register_host) into device memory
//     on its own stream, then computes: into the expert's admission reservation when prepare made one (the copy
//     is the admission; finish publishes it), else into one of `slots` device slots the tier allocates at a token
//     boundary after a batch first wanted one. A pageable source, or no free slot, refuses the batch.
// The sink places each expert from measured costs. launch queues jobs behind the layer's resident hits, after
// the layer's format launch (accepted is 0, and nothing is queued, when this layer has no launched GPU work to
// follow or the batch is refused); finish waits for them and writes each job's weighted down row to its row of the
// output, as for hits. A payload must stay valid until finish. costs reports the GPU time per expert measured so
// far for a resident hit and for a host-memory job (microseconds, 0 until a sample exists).
// register_host page-locks [pointer, pointer + bytes) and maps it for the device (cudaHostRegister); unregister_host
// undoes it, and must come before the caller unmaps or discards those pages. A copy may not span two registered
// ranges. Both return 0 and do nothing where the device reads host memory in place. Either may run on any thread;
// a registration holds the driver for about a millisecond per 19 MB and delays other threads' CUDA calls meanwhile.
struct kimi_vram_host_job { int32_t row; float weight; const void * role[3]; };
// The measured lines of the host path (microseconds; 0 until measured).
struct kimi_vram_host_costs {
    double resident_call_us, resident_expert_us; // the residents' engine call: call + expert * k
    double host_call_us, host_expert_us;         // a call of host jobs (in place, or from device memory after the copies)
    double copy_us_per_mb;                        // a page-locked copy into device memory (0: in place, or not measured)
    double link_busy_us;                          // copies already queued on the link (deferred admissions)
    int32_t in_place;                             // 1: the device reads host memory in place; no copy, no slot
    int32_t slots;                                // discrete: device slots this layer's jobs without a reservation can
                                                  // take (free scratch slots; before they are allocated, the slots a
                                                  // first batch would ask for; 0 after their allocation failed)
    uint64_t reserved_rows;                       // discrete: bit j, the layer's request j (j < 64) holds an admission
                                                  // reservation a job copies into (no scratch slot needed)
};
struct kimi_vram_host_api {
    uint32_t version, size;   // 2
    int (*available)(void * context);
    int (*launch)(void * context, const struct kimi_vram_host_job * jobs, int count, int * accepted,
                  char * error, size_t error_size);
    void (*costs)(void * context, double * resident_us, double * host_us);
    int (*register_host)(void * context, const void * pointer, size_t bytes, char * error, size_t error_size);
    int (*unregister_host)(void * context, const void * pointer, char * error, size_t error_size);
    void (*costs2)(void * context, struct kimi_vram_host_costs * costs);
};
// Optional extension, the sink's side of the tier's decisions.
// set_miss_cost: the measured cost of an expert the tier does not hold (microseconds: the slower of its
// compute and its share of the layer's SSD reads). The tier admits an expert when the hits it can expect to
// earn are worth more than its upload: selection count > upload / (hits per count x (miss cost - hit
// cost)), from its own measured upload time and hit history.
// set_host_demand: host memory the sink may still commit (its RAM cache's mapped but not yet resident
// slots). On unified memory the tier keeps it free, with the host floor, instead of taking it first.
// set_host_floor: the floor of the sink's host-memory ledger (file reserve + margin + explicit reserves). On
// unified memory the tier keeps it free and publishes floor + demand as ggml's allocation reserve.
// reclaim_host: on unified memory, room for `bytes` of host memory the ledger is about to grant outside the tier.
// The tier gives its last blocks back (keeping one) while MemAvailable less the floor and the demand is below
// bytes plus its low watermark, waiting for deferred uploads first (collect reports those in a freed block as
// aborted). 0 when that much is free; 1 when a layer is live (nothing waits or is freed) or not enough could go.
// host_bytes (version 3): on unified memory, the DRAM the tier holds (its blocks and prefill scratch); 0 on a
// discrete GPU. The ledger subtracts it from MemAvailable changes it learns its margin from.
struct kimi_vram_policy_api {
    uint32_t version, size;   // 3
    void (*set_miss_cost)(void * context, double miss_us);
    void (*set_host_demand)(void * context, uint64_t bytes);
    void (*set_host_floor)(void * context, uint64_t bytes);
    int (*reclaim_host)(void * context, uint64_t bytes);
    uint64_t (*host_bytes)(void * context);
};
#ifdef __cplusplus
extern "C" {
#endif
// The optional DSO exports this symbol. No DSO load or GPU work with budget 0.
typedef const struct kimi_vram_tier_api * (*kimi_vram_tier_get_api_fn)(void);
typedef const struct kimi_vram_residency_api * (*kimi_vram_residency_get_api_fn)(void);
typedef const struct kimi_vram_format_api * (*kimi_vram_format_get_api_fn)(void);
typedef const struct kimi_vram_prefill_api * (*kimi_vram_prefill_get_api_fn)(void);
typedef const struct kimi_vram_deferred_api * (*kimi_vram_deferred_get_api_fn)(void);
typedef const struct kimi_vram_elastic_api * (*kimi_vram_elastic_get_api_fn)(void);
typedef const struct kimi_vram_host_api * (*kimi_vram_host_get_api_fn)(void);
typedef const struct kimi_vram_policy_api * (*kimi_vram_policy_get_api_fn)(void);
const struct kimi_vram_elastic_api * kimi_vram_elastic_get_api(void);
#ifdef __cplusplus
}
#endif
