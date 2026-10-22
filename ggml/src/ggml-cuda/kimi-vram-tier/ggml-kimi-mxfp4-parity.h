#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Versioned, CUDA-header-free interface for a cache-owned MXFP4 x Q8_0 matvec
// batch (DeepSeek-V4.1 routed experts). Each job is one weight matrix; every
// row gives one float. Rows reproduce the AVX2 CPU backend's
// ggml_vec_dot_mxfp4_q8_0 bit for bit: eight exact int32 lanes of four signed
// products per 32-element block, the block scale fused (FMA) into one per-lane
// accumulator for even and another for odd blocks, their sum, then the fixed
// hsum_float_8 tree. Q8_0 inputs are those of quantize_row_q8_0 ([-127, 127]).
// The caller retains device weights until wait completes. submit copies the
// descriptors and Q8_0 bytes before returning. result is owned by the handle
// and remains valid until the next submit. A handle is not thread-safe.
struct kimi_mxfp4_gpu;
struct kimi_mxfp4_gpu_job {
    const void * weights;   // device rows of block_mxfp4
    size_t weight_bytes;
    size_t row_stride;      // bytes per row, at least columns / 32 * 17
    size_t q8_offset;       // this job's block_q8_0 row inside the submitted input
    int32_t columns;        // multiple of 64 (an even number of 32-element blocks)
    int32_t rows;
};
struct kimi_mxfp4_gpu_config {
    int32_t device;
    int32_t max_jobs;
    int32_t max_columns;
    int32_t max_rows;
};
struct kimi_mxfp4_gpu_api_v1 {
    uint32_t version;       // 1
    uint32_t job_size;
    struct kimi_mxfp4_gpu * (*create)(const struct kimi_mxfp4_gpu_config *, char *, size_t);
    int (*submit)(struct kimi_mxfp4_gpu *, const struct kimi_mxfp4_gpu_job *, int, const void *, size_t);
    int (*query)(struct kimi_mxfp4_gpu *); // 0 pending, 1 ready/idle, -1 failure
    int (*wait)(struct kimi_mxfp4_gpu *);
    const float * (*result)(struct kimi_mxfp4_gpu *, int job, int * rows);
    const char * (*error)(struct kimi_mxfp4_gpu *);
    int (*destroy)(struct kimi_mxfp4_gpu *); // synchronize, release only owned scratch
    size_t (*host_bytes)(struct kimi_mxfp4_gpu *);
    size_t (*device_bytes)(struct kimi_mxfp4_gpu *);
    // Upload helpers for the engine self-check and tests. Allocations belong to the caller.
    int (*upload)(struct kimi_mxfp4_gpu *, const void *, size_t, void **);
    int (*release_weights)(struct kimi_mxfp4_gpu *, void *);
};

#define KIMI_MXFP4_EXPORT __attribute__((visibility("default")))
KIMI_MXFP4_EXPORT const struct kimi_mxfp4_gpu_api_v1 * kimi_mxfp4_gpu_get_api_v1(void);

#ifdef __cplusplus
}
#endif
