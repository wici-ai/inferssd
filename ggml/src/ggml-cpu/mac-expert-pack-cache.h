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

#ifdef __cplusplus
}
#endif
