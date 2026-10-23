#pragma once
#include "ggml-kimi-mxfp4-parity.h"

#ifdef __cplusplus
extern "C" {
#endif

// Optional GPU hidden chain for the MXFP4 VRAM tier (DeepSeek-V4.1 routed experts),
// compiled into the same DSO as the matvec parity component: kimi-mxfp4-chain.cu
// includes private mxfp4-kernels.cuh unchanged. One submission per layer: one H2D of the
// job descriptors and the Q8_0 input, the stage-1 up/gate matvec, the clamped SwiGLU
// on the device (device expf), device Q8_0 of the hidden rows, the stage-2 down matvec
// on that device buffer and one D2H of the unweighted down rows. NOT bitwise equal to
// the CPU sink or to the host bridge; see the tier README for the measured error.
enum kimi_mxfp4_chain_kernel {
    KIMI_MXFP4_CHAIN_PARITY_Q8 = 0,  // both matvecs with the parity lane kernel, device Q8_0 hidden rows
    KIMI_MXFP4_CHAIN_ROWS_Q8 = 1,    // one thread per output row (generic block order), device Q8_0 hidden rows
    KIMI_MXFP4_CHAIN_ROWS_FLOAT = 2, // rows kernel; stage 2 reads the float hidden rows, no hidden Q8_0
};
struct kimi_mxfp4_chain_expert {
    const void * up, * gate, * down; // device rows of block_mxfp4: hidden x latent, hidden x latent, latent x hidden
};
struct kimi_mxfp4_gpu_chain;
// GPU event intervals (ms) summed over completed timed submissions of a base handle,
// plus the host time spent inside its H2D copy calls (includes any H2D pacing). The
// h2d interval runs from the submission's first mark to the end of its H2D copies, so
// it also contains host-side pacing before the copy was enqueued.
struct kimi_mxfp4_gpu_timing {
    uint64_t submits;
    double h2d_ms, stage1_ms, activation_ms, quantize_ms, stage2_ms, d2h_ms;
    double h2d_call_us;
};
struct kimi_mxfp4_gpu_chain_api_v1 {
    uint32_t version, size; // 1
    // The base handle must come from the same DSO, outlive the chain and have
    // max_jobs >= 2 * max_hits and max_columns, max_rows >= latent, hidden.
    struct kimi_mxfp4_gpu_chain * (*create)(struct kimi_mxfp4_gpu * base, int latent, int hidden, int max_hits,
                                            int kernel, int keep_hidden);
    // Enqueue one layer: experts[i] for i < hits, the layer's Q8_0 input row (latent
    // columns) and the SwiGLU limit. The base API's query/wait/error consume it and
    // result(base, i, &rows) returns hit i's unweighted down row (latent floats).
    int (*submit)(struct kimi_mxfp4_gpu_chain *, const struct kimi_mxfp4_chain_expert * experts, int hits,
                  const void * q8_input, size_t bytes, float swiglu_limit);
    // keep_hidden only (tests): the last completed submission's hidden rows, block_q8_0
    // bytes per hit, or hidden floats per hit for KIMI_MXFP4_CHAIN_ROWS_FLOAT.
    const void * (*hidden)(struct kimi_mxfp4_gpu_chain *, size_t * bytes);
    int (*destroy)(struct kimi_mxfp4_gpu_chain *); // synchronize and release the chain buffers
    size_t (*host_bytes)(struct kimi_mxfp4_gpu_chain *);
    size_t (*device_bytes)(struct kimi_mxfp4_gpu_chain *);
    // Timing of every later submission of a base handle (bridge and chain). For bridge
    // submissions, stage selects the slot of the kernel interval: 1 up/gate, 2 down.
    int (*set_timing)(struct kimi_mxfp4_gpu * base, int enabled);
    int (*set_timing_stage)(struct kimi_mxfp4_gpu * base, int stage);
    int (*timing)(struct kimi_mxfp4_gpu * base, struct kimi_mxfp4_gpu_timing * out);
};

KIMI_MXFP4_EXPORT const struct kimi_mxfp4_gpu_chain_api_v1 * kimi_mxfp4_gpu_get_chain_api_v1(void);

#ifdef __cplusplus
}
#endif
