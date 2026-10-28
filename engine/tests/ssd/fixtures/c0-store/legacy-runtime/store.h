#pragma once
// Experimental store link boundary; not a stable public ABI.
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// The process-fixed immutable store owns the returned descriptor. Consumers
// borrow it and must finish their reads before process teardown; never close it.
// Offsets and lengths remain logical. The sink retains its existing alignment.
bool ggml_ssd_c0_resolve_raw_direct_shape(
        const char * tensor_name, int expert, size_t bytes, int64_t experts,
        int * fd, uint64_t * offset, size_t * logical_bytes);

// Same first-shape state for CPU and CUDA adapter entry points.
void ggml_ssd_c0_select_model_shape(int64_t experts, size_t bytes, const char * source);
bool ggml_ssd_c0_uses_common();
bool ggml_ssd_c0_shape_selected();

#ifdef __cplusplus
}
#endif
