#pragma once
#include "ggml-kimi-q3k-parity.h"
#ifdef __cplusplus
extern "C" {
#endif
struct kimi_q3k_gpu_chain;
// Optional GPU SiTU/Q8_K chain for the exact K3 3584/3072 shape. The base
// handle must be created by the same DSO and outlive this chain. Its existing
// wait/result/error API consumes chain submissions. No host intermediate copy.
struct kimi_q3k_gpu_chain_api_v1 {
 uint32_t version,size;
 struct kimi_q3k_gpu_chain * (*create)(struct kimi_q3k_gpu *);
 int (*submit)(struct kimi_q3k_gpu_chain *,const struct kimi_q3k_gpu_job * up_gate,
               const struct kimi_q3k_gpu_job * down,int pairs,const void * q8,size_t bytes,float beta,float linear);
 int (*destroy)(struct kimi_q3k_gpu_chain *);
 size_t (*host_bytes)(struct kimi_q3k_gpu_chain *);
 size_t (*device_bytes)(struct kimi_q3k_gpu_chain *);
};
KIMI_Q3K_EXPORT const struct kimi_q3k_gpu_chain_api_v1 * kimi_q3k_gpu_get_chain_api_v1(void);
#ifdef __cplusplus
}
#endif
