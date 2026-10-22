#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct kimi_mxfp4_batch;

struct kimi_mxfp4_batch_api {
    uint32_t version, size;
    struct kimi_mxfp4_batch * (*create)(int device, int latent, int hidden, int capacity, char * error, size_t error_size);
    int (*compute)(struct kimi_mxfp4_batch *, const void * up, const void * gate, const void * down,
                   const void * q8_input, int tokens, float swiglu_limit, float * output,
                   char * error, size_t error_size);
    int (*destroy)(struct kimi_mxfp4_batch *);
    size_t (*device_bytes)(struct kimi_mxfp4_batch *);
    size_t (*host_bytes)(struct kimi_mxfp4_batch *);
    int (*begin_accumulate)(struct kimi_mxfp4_batch *, int tokens, char * error, size_t error_size);
    int (*compute_accumulate)(struct kimi_mxfp4_batch *, const void * up, const void * gate, const void * down,
                              const void * q8_input, const int * token_indices, const float * weights,
                              int routes, float swiglu_limit, char * error, size_t error_size);
    int (*finish_accumulate)(struct kimi_mxfp4_batch *, const int * token_indices, int count,
                             float * output, char * error, size_t error_size);
    int (*release_accumulate)(struct kimi_mxfp4_batch *, char * error, size_t error_size);
    uint64_t (*h2d_bytes)(struct kimi_mxfp4_batch *);
    uint64_t (*d2h_bytes)(struct kimi_mxfp4_batch *);
};

const struct kimi_mxfp4_batch_api * kimi_mxfp4_batch_get_api(void);

#ifdef __cplusplus
}
#endif
