#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
// Free memory a new device allocation can get: on a unified-memory GPU, where device allocations come out of
// system memory and cudaMemGetInfo reports only MemFree, MemAvailable as it reads at the kernel's full
// watermark boost (MemAvailable keeps the boosted low watermarks back).
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);
GGML_BACKEND_API bool ggml_backend_cuda_device_is_unified(int device);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

// Optional phase-scoped recovery for direct backend-buffer cudaMalloc OOMs.
// A callback returns 1 after freeing one block, 0 when nothing is reclaimable,
// and -1 on a fail-closed release error. Registration is exclusive per device.
typedef int (*ggml_cuda_oom_reclaimer_fn)(void * userdata, size_t requested_bytes, size_t * freed_bytes);
GGML_BACKEND_API bool ggml_cuda_oom_reclaimer_register(
        int device, ggml_cuda_oom_reclaimer_fn callback, void * userdata);
GGML_BACKEND_API bool ggml_cuda_oom_reclaimer_unregister(
        int device, ggml_cuda_oom_reclaimer_fn callback, void * userdata);
// Memory that device allocations must leave free, set at run time by a cache that can give memory back
// through the OOM reclaimer (on unified memory, the host's own working set). The effective reserve is
// the larger of this and GGML_CUDA_MEMORY_RESERVE_MIB.
GGML_BACKEND_API void ggml_cuda_set_memory_reserve(int device, size_t bytes);
// Pinned host memory about to be allocated for the current device: on a unified-memory GPU it comes out of the same
// DRAM, so the reserve is made first (the OOM reclaimer gives memory back). False when no room could be made; the
// caller may allocate anyway. True on a discrete GPU.
GGML_BACKEND_API bool ggml_cuda_admit_host_alloc(size_t size);
// Deterministic test hook; the next count backend-buffer allocations report
// OOM before cudaMalloc and exercise the same reclaim-and-retry loop.
GGML_BACKEND_API bool ggml_cuda_oom_reclaimer_inject_failures(int device, uint64_t count);

// Optional prefill seeding of an expert cache (one owner per device). The scheduler offers every routed expert
// tensor it copied to a device for a graph pass (ggml_cuda_moe_seed_offer): the tensor's name, its device copy,
// the bytes of one expert, the expert count and the pass's routing count of each expert, with `ready` (a
// cudaEvent_t) recorded on the backend's stream behind the copy. The callback may copy experts out of it on its
// own stream after waiting for `ready`; it sets *done to an event recorded behind those copies (or leaves it null),
// and the backend's stream waits for it before the split that copied computes (ggml_cuda_moe_seed_fence): a node
// after the expert matmul may reuse the copy's memory.
typedef int (*ggml_cuda_moe_seed_fn)(void * userdata, const char * tensor_name, const void * data, size_t expert_bytes,
        int64_t n_expert, const int32_t * counts, void * ready, void ** done);
GGML_BACKEND_API bool ggml_cuda_moe_seed_register(int device, ggml_cuda_moe_seed_fn callback, void * userdata);
GGML_BACKEND_API bool ggml_cuda_moe_seed_unregister(int device, ggml_cuda_moe_seed_fn callback, void * userdata);
GGML_BACKEND_API void ggml_cuda_moe_seed_offer(ggml_backend_t backend, const struct ggml_tensor * tensor,
        const char * tensor_name, int64_t n_expert, size_t expert_bytes, const int32_t * counts);
GGML_BACKEND_API void ggml_cuda_moe_seed_fence(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
