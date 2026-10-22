#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Versioned, CUDA-header-free interface for a cache-owned Q3_K matvec batch.
// Each job is one independent row of activations. No slot-order reduction.
// The caller retains device weights until query/wait completes. submit copies
// the host descriptors and Q8_K bytes before returning. result is owned by the
// handle and remains valid until the next submit. A handle is not thread-safe.
struct kimi_q3k_gpu;
struct kimi_q3k_gpu_job {
    const void * weights;
    size_t weight_bytes;
    size_t row_stride;
    size_t q8_offset;
    int32_t columns; // multiple of 256
    int32_t rows;
    float scale;
    int32_t apply_scale; // separate RN multiply after dot, never fused with sum
};
struct kimi_q3k_gpu_config {
    int32_t device;
    int32_t max_jobs;
    int32_t max_columns;
    int32_t max_rows;
};
struct kimi_q3k_gpu_api_v1 {
    uint32_t version;
    uint32_t job_size;
    struct kimi_q3k_gpu * (*create)(const struct kimi_q3k_gpu_config *, char *, size_t);
    int (*submit)(struct kimi_q3k_gpu *, const struct kimi_q3k_gpu_job *, int, const void *, size_t);
    int (*query)(struct kimi_q3k_gpu *); // 0 pending, 1 ready/idle, -1 failure
    int (*wait)(struct kimi_q3k_gpu *);
    const float * (*result)(struct kimi_q3k_gpu *, int job, int * rows);
    const char * (*error)(struct kimi_q3k_gpu *);
    int (*destroy)(struct kimi_q3k_gpu *); // synchronize, release only owned scratch
    size_t (*host_bytes)(struct kimi_q3k_gpu *);
    size_t (*device_bytes)(struct kimi_q3k_gpu *);
    // Optional upload helpers. Allocations belong to caller, NOT the handle.
    int (*upload)(struct kimi_q3k_gpu *, const void *, size_t, void **);
    int (*release_weights)(struct kimi_q3k_gpu *, void *);
};

#if defined(_WIN32)
#define KIMI_Q3K_EXPORT __declspec(dllexport)
#else
#define KIMI_Q3K_EXPORT __attribute__((visibility("default")))
#endif
KIMI_Q3K_EXPORT const struct kimi_q3k_gpu_api_v1 * kimi_q3k_gpu_get_api_v1(void);

#ifdef __cplusplus
}
#endif
