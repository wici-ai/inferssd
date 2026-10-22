#pragma once
#include "kimi-cpu-sink.h"

// Executor-facing view of the ONE registry owned by the RAM/VRAM placement
// provider. This adapter does not implement a second cache or CUDA math path.
// The caller serializes attach/begin/run/reset/destroy. All callbacks run inside
// sink run_mutex and must not reenter sink APIs. Context/DSO outlives the sink.
struct kimi_cache_location_candidate {
    uint64_t epoch, token;
    int32_t layer, expert;
    struct kimi_cpu_sink_source role[3];
};
struct kimi_cache_location_lease {
    uint64_t handle, generation; // handle=0 means not VRAM READY
};
struct kimi_cache_location_job {
    struct kimi_cache_location_lease lease;
    int32_t native_slot;
    float * raw_down; // latent F32, UNWEIGHTED; owned until finish/cancel
};
struct kimi_cache_location_api {
    uint32_t version; // 1
    uint32_t size;
    // Source identities must be validated (device/inode/size/timestamps/offset),
    // epoch checked, READY acquired atomically. Upload-pending is not a hit.
    // Query/pin must not train frequency or start admission. On failure any
    // written leases remain valid for cancel+release to drain.
    int (*acquire_native)(void *, const struct kimi_cache_location_candidate *, int,
                          struct kimi_cache_location_lease *, char *, size_t);
    // base_mask is actual ordinary demand; pin_mask includes cache-only extras.
    // No top16-wide frequency training. Called after unused leases are released.
    int (*begin_layer)(void *, const struct kimi_cache_location_candidate *, int,
                       uint32_t base_mask, uint32_t pin_mask, char *, size_t);
    // One selected batch per layer. Copies Q8_K input and job descriptors before returning. Jobs
    // contain only final selected VRAM entries, never the unused reserve pool.
    int (*launch)(void *, const struct kimi_cache_location_job *, int,
                  const void * input_q8k, size_t input_bytes, int latent, int hidden,
                  float beta, float linear_beta, char *, size_t);
    // Optional ordinary-demand admission. borrow=1 retains RAM through finish;
    // borrow=0 must copy a ring source before return. No extra SSD reads.
    int (*admit)(void *, const struct kimi_cache_location_candidate *,
                 const void * up, const void * gate, const void * down, int borrow,
                 char *, size_t);
    // Both are required and drain all owned GPU compute AND uploads. Idempotent
    // cancel after an error/finish must be safe; failure to drain is fatal.
    int (*finish)(void *, char *, size_t);
    int (*cancel)(void *, char *, size_t);
    int (*release)(void *, const struct kimi_cache_location_lease *, int, char *, size_t);
    // Optional shared-registry eviction callback, after GPU/IO/CPU/reduction and
    // every native lease is released. RAM owner applies confirmed drops here;
    // preserve touched/committed bytes. No callback while a payload is in use.
    int (*finished)(void *, struct kimi_cpu_sink *, char *, size_t);
};
