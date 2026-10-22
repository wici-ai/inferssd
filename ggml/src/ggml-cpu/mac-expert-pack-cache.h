#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opt-in macOS expert-pack reader. A successful acquire pins the returned
// slice until release, so all ggml worker threads may read it safely.
bool ggml_mac_expert_pack_enabled(void);
const void * ggml_mac_expert_pack_acquire(
        const char * tensor_name, int32_t expert, size_t logical_bytes);
void ggml_mac_expert_pack_release(const void * data);

// Read ahead only the selected experts in one op, without changing arithmetic
// or cache admission order. Every selected expert must be acquired once in
// ascending order; end_batch drains the bounded read queue before returning.
void * ggml_mac_expert_pack_begin_batch(const char * name, const int64_t * counts, int n, size_t bytes);
const void * ggml_mac_expert_pack_acquire_batch(void * batch, const char * name, int32_t expert, size_t bytes);
void ggml_mac_expert_pack_end_batch(void * batch);
void ggml_mac_expert_pack_record_compute(uint64_t us, uint64_t barriers);

#ifdef __cplusplus
}
#endif
