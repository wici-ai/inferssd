#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Context-owned macOS expert-pack reader. Null cache disables the hook. A successful acquire pins the returned
// slice until release, so all ggml worker threads may read it safely.
bool ggml_mac_expert_pack_enabled(void * cache);
bool ggml_mac_expert_pack_prefill_enabled(void * cache);
const void * ggml_mac_expert_pack_acquire(
        void * cache, const char * tensor_name, int32_t expert, size_t logical_bytes);
void ggml_mac_expert_pack_release(void * cache, const void * data);

// Read ahead only the selected experts in one op, without changing arithmetic
// or cache admission order. Every selected expert must be acquired once in
// ascending order; end_batch drains the bounded read queue before returning.
// Prefill may reuse cached bytes without changing the decode cache's admission,
// heat or recency. Misses then live only in bounded transient read buffers.
void * ggml_mac_expert_pack_begin_batch(void * cache, const char * name, const int64_t * counts, int n, size_t bytes, bool cache_updates);
const void * ggml_mac_expert_pack_acquire_batch(void * cache, void * batch, const char * name, int32_t expert, size_t bytes);
void ggml_mac_expert_pack_end_batch(void * cache, void * batch);
void ggml_mac_expert_pack_record_compute(void * cache, uint64_t us, uint64_t barriers);

#ifdef __cplusplus
}
#endif
