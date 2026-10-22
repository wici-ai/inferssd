#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "traits.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "quants.h"
#include "ggml-threading.h"
#include "unary-ops.h"
#include "binary-ops.h"
#include "vec.h"
#include "ops.h"
#include "ggml.h"
#include "common.h"
#if defined(__APPLE__)
#include "mac-expert-pack-cache.h"
#endif

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h> // using malloc.h with MSC/MINGW
#elif !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#include <alloca.h>
#endif

#include <assert.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <stdarg.h>
#include <signal.h>
#if defined(__gnu_linux__)
#include <syscall.h>
#include <sys/mman.h>
#endif

#ifdef GGML_USE_OPENMP
#include <omp.h>
#endif

#if defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#undef GGML_USE_LLAMAFILE
#endif

#ifdef GGML_USE_LLAMAFILE
#include "llamafile/sgemm.h"
#endif

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#define GGML_CACHE_LINE  64

#if defined(__clang__) || defined(__GNUC__)
#define GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_LINE)))
#endif

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#else  // __has_feature
#if defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif // __has_feature

#define UNUSED GGML_UNUSED
#define SWAP(x, y, T) do { T SWAP = x; (x) = y; (y) = SWAP; } while (0)

typedef struct { int32_t i1; int32_t i2; } ggml_moe_stream_row_mapping;
typedef struct {
    void * bank_ptr;
    size_t slot_stride;
    size_t ring_bytes;
    uint64_t logical_bytes;
    uint64_t physical_read_bytes;
    uint64_t stage_us;
    uint64_t iouring_reads;
    uint64_t ram_admissions;
    uint64_t vram_admissions;
    uint64_t direct_fallbacks;
    int32_t bank_index;
    int32_t n_experts;
} ggml_moe_cpu_bank_result;
typedef bool (*ggml_moe_cpu_wave_ready_fn)(void *, int, const ggml_moe_cpu_bank_result *);
typedef bool (*ggml_moe_cpu_wave_released_fn)(void *, int, bool);

extern bool ggml_cpu_layer_gap_trace_enabled(void);
extern void ggml_cpu_layer_gap_trace_node(
    uint64_t start_us,
    uint64_t end_us,
    int op,
    const char * op_name,
    const char * name);

#if !defined(__APPLE__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((weak)) extern bool ggml_cuda_kimi_dense_stream_mul_mat(
    int src0_type_int,
    const char * src0_name,
    const void * src0_data,
    size_t src0_bytes,
    int64_t ne00,
    int64_t ne01,
    int64_t ne02,
    int64_t ne03,
    size_t nb01,
    size_t nb02,
    size_t nb03,
    const float * src1_data,
    int64_t ne11,
    int64_t ne12,
    int64_t ne13,
    size_t nb11,
    size_t nb12,
    size_t nb13,
    float * dst_data,
    size_t nb1,
    size_t nb2,
    size_t nb3);
__attribute__((weak)) extern bool ggml_cuda_kimi_dense_stream_should_intercept(const char * tensor_name);
__attribute__((weak)) extern void ggml_cuda_moe_adapter_select_model_shape(
    int64_t n_experts,
    size_t expert_bytes);
__attribute__((weak)) extern bool ggml_cuda_moe_stream_available(void);
__attribute__((weak)) extern void ggml_cuda_moe_stream_sync(void);
__attribute__((weak)) extern bool ggml_cuda_moe_stream_one(
    int src0_type_int,
    const char * src0_name,
    int64_t expert_index,
    const void * src0_data,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    const float * src1_f32,
    size_t src1_nb1,
    size_t src1_nb2,
    int64_t src1_ne1,
    int64_t cne1,
    const void * src1_q8_1,
    size_t src1_padded_num_cols,
    float * dst,
    size_t dst_nb1,
    size_t dst_nb2,
    const ggml_moe_stream_row_mapping * rows);
__attribute__((weak)) extern void ggml_cuda_moe_stream_q80_probe(
    int src0_type_int,
    const char * src0_name,
    int64_t expert_index,
    const void * src0_data,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    const void * src1_q8_0,
    size_t src1_q8_0_row_size,
    int64_t src1_ne1,
    int64_t cne1,
    const float * dst,
    size_t dst_nb1,
    size_t dst_nb2,
    const ggml_moe_stream_row_mapping * rows);
__attribute__((weak)) extern void ggml_cuda_moe_stream_q80_write(
    int src0_type_int,
    const char * src0_name,
    int64_t expert_index,
    const void * src0_data,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    const void * src1_q8_0,
    size_t src1_q8_0_row_size,
    int64_t src1_ne1,
    int64_t cne1,
    float * dst,
    size_t dst_nb1,
    size_t dst_nb2,
    const ggml_moe_stream_row_mapping * rows);
__attribute__((weak)) extern bool ggml_cuda_moe_stream_q80_skip(
    int src0_type_int,
    const char * src0_name,
    int64_t expert_index,
    const void * src0_data,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    const void * src1_q8_0,
    size_t src1_q8_0_row_size,
    int64_t src1_ne1,
    int64_t cne1,
    float * dst,
    size_t dst_nb1,
    size_t dst_nb2,
    const ggml_moe_stream_row_mapping * rows);
__attribute__((weak)) extern void ggml_cuda_moe_stream_q80_hot_batch_probe(
    int src0_type_int,
    const char * src0_name,
    int64_t n_as,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    const void * src1_q8_0,
    size_t src1_q8_0_row_size,
    int64_t src1_ne1,
    const float * src1_f32,
    size_t src1_nb1,
    size_t src1_nb2,
    const int64_t * matrix_row_counts,
    const ggml_moe_stream_row_mapping * matrix_rows,
    int64_t rows_per_expert,
    const float * dst,
    size_t dst_nb1,
    size_t dst_nb2);
__attribute__((weak)) extern bool ggml_cuda_moe_stream_handoff_upload(
    const float * host_ptr,
    int64_t ne01,
    int64_t dst_cols);
__attribute__((weak)) extern bool ggml_cuda_moe_stream_batch(
    int src0_type_int,
    const char * src0_name,
    const void * src0_data,
    int64_t n_as,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    size_t nb02,
    const float * src1_f32,
    int64_t src1_ne1,
    size_t src1_nb1,
    size_t src1_nb2,
    const void * src1_q8_0,
    size_t src1_q8_0_row_size,
    int64_t src1_q8_0_ne1,
    float * dst,
    size_t dst_nb1,
    size_t dst_nb2,
    const int64_t * matrix_row_counts,
    const ggml_moe_stream_row_mapping * rows,
    int64_t rows_stride);
__attribute__((weak)) extern bool ggml_cuda_moe_a2_read_pipeline(
    const char * tensor_name,
    const int32_t * expert_ids,
    const int32_t * wave_sizes,
    int n_waves,
    size_t expert_bytes,
    void * opaque,
    ggml_moe_cpu_wave_ready_fn wave_ready,
    ggml_moe_cpu_wave_released_fn wave_released);
__attribute__((weak)) extern bool ggml_cuda_moe_stream_decode_batch(
    int src0_type_int,
    const char * src0_name,
    const void * src0_data,
    int64_t n_as,
    int64_t ne01,
    int64_t ne00,
    size_t nb01,
    size_t nb02,
    const float * src1_f32,
    int64_t src1_ne1,
    size_t src1_nb1,
    size_t src1_nb2,
    const void * src1_q8_0,
    size_t src1_q8_0_row_size,
    float * dst,
    size_t dst_nb1,
    size_t dst_nb2,
    int64_t * matrix_row_counts,
    const ggml_moe_stream_row_mapping * rows,
    int64_t rows_stride);
__attribute__((weak)) extern const void * ggml_cuda_moe_stream_ds4_triplet_host_role_ptr(
    const char * src0_name,
    int64_t expert_index,
    size_t role_nbytes);
__attribute__((weak)) extern int ggml_cuda_moe_stream_batch_preload_active_from_pack(
    int src0_type_int,
    const char * src0_name,
    int64_t n_as,
    size_t expert_bytes,
    const int64_t * matrix_row_counts);
__attribute__((weak)) extern int ggml_cuda_moe_stream_q4_colibri_pilot_prefetch_from_gate(
    int src0_type_int,
    const char * src0_gate_name,
    int64_t n_as,
    size_t expert_bytes,
    const int64_t * matrix_row_counts);
__attribute__((weak)) extern int ggml_cuda_moe_stream_route_group_preload_updown_from_gate(
    const char * src0_gate_name,
    int64_t n_as,
    const int64_t * matrix_row_counts);
__attribute__((weak)) extern const void * ggml_cuda_moe_expert_pack_mmap_ptr(
    const char * tensor_name,
    int expert_idx,
    size_t nbytes);
__attribute__((weak)) extern const void * ggml_cuda_moe_expert_pack_mmap_ptr_debug(
    const char * tensor_name,
    int expert_idx,
    size_t nbytes,
    const char ** reason,
    size_t * entry_nbytes,
    uint64_t * entry_offset);
__attribute__((weak)) extern const void * ggml_cuda_moe_expert_pack_mmap_ptr_source_debug(
    const char * tensor_name,
    int expert_idx,
    size_t nbytes,
    const char ** reason,
    size_t * entry_nbytes,
    uint64_t * entry_offset,
    int * source_idx,
    const char ** source_path,
    int * source_is_gguf);
__attribute__((weak)) extern bool ggml_cuda_moe_stream_up_gate_batch(
    int src0_up_type_int,
    int src0_gate_type_int,
    const char * src0_up_name,
    const void * src0_up_data,
    const char * src0_gate_name,
    const void * src0_gate_data,
    int64_t n_as,
    int64_t ne01,
    int64_t ne00,
    size_t up_nb01,
    size_t up_nb02,
    size_t up_expert_bytes,
    size_t gate_nb01,
    size_t gate_nb02,
    size_t gate_expert_bytes,
    const float * src1_f32,
    size_t src1_nb1,
    size_t src1_nb2,
    float * dst,
    size_t dst_nb1,
    size_t dst_nb2,
    int op,
    float limit,
    const int64_t * matrix_row_counts,
    const ggml_moe_stream_row_mapping * rows,
    int64_t rows_stride);
__attribute__((weak)) extern bool ggml_cuda_moe_stream_cache_contains(
    const char * src0_name,
    size_t expert_bytes,
    int expert_idx);
__attribute__((weak)) extern bool ggml_cuda_moe_stream_one_cache_contains(
    const char * src0_name,
    const void * src0_data,
    size_t expert_bytes,
    int64_t expert_idx);
#else
static bool (*ggml_cuda_kimi_dense_stream_mul_mat)(
    int, const char *, const void *, size_t,
    int64_t, int64_t, int64_t, int64_t, size_t, size_t, size_t,
    const float *, int64_t, int64_t, int64_t, size_t, size_t, size_t,
    float *, size_t, size_t, size_t) = NULL;
static bool (*ggml_cuda_kimi_dense_stream_should_intercept)(const char *) = NULL;
static void (*ggml_cuda_moe_adapter_select_model_shape)(int64_t, size_t) = NULL;
static bool (*ggml_cuda_moe_stream_available)(void) = NULL;
static void (*ggml_cuda_moe_stream_sync)(void) = NULL;
static bool (*ggml_cuda_moe_stream_one)(
    int, const char *, int64_t, const void *, int64_t, int64_t, size_t, const float *, size_t, size_t,
    int64_t, int64_t, const void *, size_t, float *, size_t, size_t,
    const ggml_moe_stream_row_mapping *) = NULL;
static void (*ggml_cuda_moe_stream_q80_probe)(
    int, const char *, int64_t, const void *, int64_t, int64_t, size_t, const void *, size_t,
    int64_t, int64_t, const float *, size_t, size_t, const ggml_moe_stream_row_mapping *) = NULL;
static void (*ggml_cuda_moe_stream_q80_write)(
    int, const char *, int64_t, const void *, int64_t, int64_t, size_t, const void *, size_t,
    int64_t, int64_t, float *, size_t, size_t, const ggml_moe_stream_row_mapping *) = NULL;
static bool (*ggml_cuda_moe_stream_q80_skip)(
    int, const char *, int64_t, const void *, int64_t, int64_t, size_t, const void *, size_t,
    int64_t, int64_t, float *, size_t, size_t, const ggml_moe_stream_row_mapping *) = NULL;
static void (*ggml_cuda_moe_stream_q80_hot_batch_probe)(
    int, const char *, int64_t, int64_t, int64_t, size_t, const void *, size_t, int64_t,
    const float *, size_t, size_t, const int64_t *, const ggml_moe_stream_row_mapping *,
    int64_t, const float *, size_t, size_t) = NULL;
static bool (*ggml_cuda_moe_stream_batch)(
    int, const char *, const void *, int64_t, int64_t, int64_t, size_t, size_t,
    const float *, int64_t, size_t, size_t, const void *, size_t, int64_t, float *, size_t, size_t,
    const int64_t *, const ggml_moe_stream_row_mapping *, int64_t) = NULL;
static bool (*ggml_cuda_moe_a2_read_pipeline)(
    const char *, const int32_t *, const int32_t *, int, size_t, void *,
    ggml_moe_cpu_wave_ready_fn, ggml_moe_cpu_wave_released_fn) = NULL;
static bool (*ggml_cuda_moe_stream_decode_batch)(
    int, const char *, const void *, int64_t, int64_t, int64_t, size_t, size_t,
    const float *, int64_t, size_t, size_t, const void *, size_t, float *, size_t, size_t,
    int64_t *, const ggml_moe_stream_row_mapping *, int64_t) = NULL;
static const void * (*ggml_cuda_moe_stream_ds4_triplet_host_role_ptr)(
    const char *, int64_t, size_t) = NULL;
static int (*ggml_cuda_moe_stream_batch_preload_active_from_pack)(
    int, const char *, int64_t, size_t, const int64_t *) = NULL;
static int (*ggml_cuda_moe_stream_q4_colibri_pilot_prefetch_from_gate)(
    int, const char *, int64_t, size_t, const int64_t *) = NULL;
static int (*ggml_cuda_moe_stream_route_group_preload_updown_from_gate)(
    const char *, int64_t, const int64_t *) = NULL;
static const void * (*ggml_cuda_moe_expert_pack_mmap_ptr)(
    const char *, int, size_t) = NULL;
static const void * (*ggml_cuda_moe_expert_pack_mmap_ptr_debug)(
    const char *, int, size_t, const char **, size_t *, uint64_t *) = NULL;
static const void * (*ggml_cuda_moe_expert_pack_mmap_ptr_source_debug)(
    const char *, int, size_t, const char **, size_t *, uint64_t *, int *, const char **, int *) = NULL;
static bool (*ggml_cuda_moe_stream_handoff_upload)(const float *, int64_t, int64_t) = NULL;
static bool (*ggml_cuda_moe_stream_up_gate_batch)(
    int, int, const char *, const void *, const char *, const void *, int64_t,
    int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, const float *, size_t, size_t, float *,
    size_t, size_t, int, float, const int64_t *, const ggml_moe_stream_row_mapping *,
    int64_t) = NULL;
static bool (*ggml_cuda_moe_stream_cache_contains)(const char *, size_t, int) = NULL;
static bool (*ggml_cuda_moe_stream_one_cache_contains)(const char *, const void *, size_t, int64_t) = NULL;
#endif

static bool ggml_kimi_dense_stream_layer_selected(const char * tensor_name) {
    const char * layers = getenv("GGML_KIMI_DENSE_STREAM_LAYERS");
    if (!layers || !layers[0] || !tensor_name) {
        return false;
    }

    if (strcmp(tensor_name, "output.weight") == 0) {
        const char * current = layers;
        while (*current) {
            while (*current == ' ' || *current == '\t' || *current == ',') {
                ++current;
            }
            const char * end = current;
            while (*end && *end != ',') {
                ++end;
            }
            const char * trimmed = end;
            while (trimmed > current && (trimmed[-1] == ' ' || trimmed[-1] == '\t')) {
                --trimmed;
            }
            const size_t length = (size_t) (trimmed - current);
            if ((length == 6 && strncmp(current, "output", length) == 0) ||
                    (length == 13 && strncmp(current, "output.weight", length) == 0)) {
                return true;
            }
            current = *end ? end + 1 : end;
        }
        return false;
    }

    if (strncmp(tensor_name, "blk.", 4) != 0) {
        return false;
    }

    char * tensor_end = NULL;
    const long tensor_layer = strtol(tensor_name + 4, &tensor_end, 10);
    if (tensor_end == tensor_name + 4 || !tensor_end || *tensor_end != '.' || tensor_layer < 0) {
        return false;
    }

    const char * current = layers;
    while (*current) {
        while (*current == ' ' || *current == '\t' || *current == ',') {
            ++current;
        }
        if (!*current) {
            break;
        }

        const char * number = current;
        if (strncmp(number, "blk.", 4) == 0) {
            number += 4;
        }
        char * end = NULL;
        const long selected = strtol(number, &end, 10);
        if (end != number) {
            while (*end == ' ' || *end == '\t') {
                ++end;
            }
            if ((*end == ',' || *end == '\0') && selected == tensor_layer) {
                return true;
            }
        }

        const char * comma = strchr(current, ',');
        if (!comma) {
            break;
        }
        current = comma + 1;
    }
    return false;
}

static bool ggml_cuda_moe_stream_supports_type(enum ggml_type type) {
    return type == GGML_TYPE_IQ1_S ||
           type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S ||
           type == GGML_TYPE_IQ2_S ||
           type == GGML_TYPE_Q2_K ||
           type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_K ||
           type == GGML_TYPE_MXFP4 || type == GGML_TYPE_F8_E4M3_B128;
}

static bool ggml_moe_gate_batch_prefetch_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = getenv("GGML_MOE_GATE_BATCH_PREFETCH");
        enabled = (env && env[0] && env[0] != '0') ? 1 : 0;
    }
    return enabled != 0;
}

static bool ggml_cuda_moe_stream_one_q4k_enabled(void) {
    const char * env = getenv("GGML_MOE_STREAM_ONE_Q4K");
    return env && env[0] && env[0] != '0';
}

static bool ggml_cuda_moe_stream_supports_one_type(enum ggml_type type) {
    const char * q8 = getenv("GGML_MOE_STREAM_Q8_0");
    const char * floating = getenv("GGML_MOE_STREAM_FLOAT");
    return ((type == GGML_TYPE_BF16 || type == GGML_TYPE_F16) && floating && floating[0] && floating[0] != '0') ||
           (type == GGML_TYPE_Q8_0 && q8 && q8[0] && q8[0] != '0') ||
           ggml_cuda_moe_stream_supports_type(type) ||
           (type == GGML_TYPE_Q4_K && ggml_cuda_moe_stream_one_q4k_enabled());
}


static bool ggml_cuda_moe_stream_supports_down_batch(enum ggml_type type, const char * name) {
    const bool is_down = name && strstr(name, "ffn_down_exps");
    const bool is_up = name && strstr(name, "ffn_up_exps");
    const bool is_gate = name && strstr(name, "ffn_gate_exps");
    // if (strstr(name,"gate") && type!=39)
    // printf("%s,%d ",name,type);
    if (is_up && type == GGML_TYPE_MXFP4) {
        const char * env = getenv("GGML_MOE_STREAM_UP_Q80_COMPAT_BATCH");
        if (!env || !env[0] || env[0] == '0') {
            return false;
        }
        const char * target = getenv("GGML_MOE_STREAM_UP_Q80_COMPAT_TENSOR");
        return !target || !target[0] || strcmp(target, name) == 0;
    }

    const char * prompt_matmul_roles = getenv("GGML_MOE_PROMPT_MATMUL_ID_BATCH_ROLES");
    const bool prompt_matmul_role_allowed =
        !prompt_matmul_roles || !prompt_matmul_roles[0] ||
        strcmp(prompt_matmul_roles, "all") == 0 ||
        (is_up && strstr(prompt_matmul_roles, "up")) ||
        (is_gate && strstr(prompt_matmul_roles, "gate"));
    const char * prompt_matmul_env = getenv("GGML_MOE_PROMPT_MATMUL_ID_BATCH");
    const bool prompt_matmul_id =
        (is_up || is_gate) &&
        prompt_matmul_role_allowed &&
        prompt_matmul_env != NULL &&
        prompt_matmul_env[0] != '0';
    if (!is_down && !prompt_matmul_id) {
        return false;
    }

    if (is_down && type == GGML_TYPE_Q4_0) {
        const char * batch = getenv("GGML_MOE_Q4_DOWN_BATCH");
        if (batch && batch[0] && batch[0] != '0') {
            const char * target = getenv("GGML_MOE_Q4_DOWN_BATCH_TENSOR");
            if (!target || !target[0] || strcmp(target, name) == 0) {
                return true;
            }
        }
        const char * route_profile = getenv("GGML_MOE_Q4_DOWN_ROUTE_PROFILE_OUT");
        if (route_profile && route_profile[0]) {
            const char * target = getenv("GGML_MOE_Q4_DOWN_ROUTE_PROFILE_TENSOR");
            if (!target || !target[0] || strcmp(target, name) == 0) {
                return true;
            }
        }

        const char * env = getenv("GGML_MOE_Q4_DOWN_PARITY");
        if (!env || !env[0] || env[0] == '0') {
            return false;
        }
        const char * target = getenv("GGML_MOE_Q4_DOWN_PARITY_TENSOR");
        return !target || !target[0] || strcmp(target, name) == 0;
    }

    return ggml_cuda_moe_stream_supports_type(type) ||
           type == GGML_TYPE_Q3_K || type == GGML_TYPE_IQ4_XS;
}


#define GGML_MOE_FALLBACK_NAME_LEN 96
#define GGML_MOE_FALLBACK_REASON_PROFILE_MAX 32768
#define GGML_MOE_FALLBACK_REASON_LEN 64

struct ggml_moe_fallback_reason_profile_entry {
    char tensor[GGML_MOE_FALLBACK_NAME_LEN];
    char role[16];
    char phase[8];
    char batch_reason[GGML_MOE_FALLBACK_REASON_LEN];
    char single_reason[GGML_MOE_FALLBACK_REASON_LEN];
    char final_reason[GGML_MOE_FALLBACK_REASON_LEN];
    int expert_idx;
    int src0_type;
    size_t expert_bytes;
    uint64_t rows;
    uint64_t calls;
    uint64_t fallback_us;
    uint64_t batch_attempts;
    uint64_t batch_accepts;
    uint64_t single_attempts;
    uint64_t single_accepts;
};

struct ggml_moe_fallback_reason_profile_state {
    bool initialized;
    bool registered;
    bool enabled;
    const char * out;
    struct ggml_moe_fallback_reason_profile_entry entries[GGML_MOE_FALLBACK_REASON_PROFILE_MAX];
    int n_entries;
    uint64_t dropped;
};

static struct ggml_moe_fallback_reason_profile_state ggml_moe_fallback_reason_profile;

#define GGML_MOE_FALLBACK_SOURCE_PROBE_MAX 32768
#define GGML_MOE_FALLBACK_SOURCE_PROBE_NAME_LEN 96
#define GGML_MOE_FALLBACK_SOURCE_PROBE_BUFT_LEN 64

struct ggml_moe_fallback_source_probe_entry {
    char tensor[GGML_MOE_FALLBACK_SOURCE_PROBE_NAME_LEN];
    char role[16];
    char phase[8];
    char batch_reason[GGML_MOE_FALLBACK_REASON_LEN];
    char single_reason[GGML_MOE_FALLBACK_REASON_LEN];
    char final_reason[GGML_MOE_FALLBACK_REASON_LEN];
    char src0_buft[GGML_MOE_FALLBACK_SOURCE_PROBE_BUFT_LEN];
    char src1_buft[GGML_MOE_FALLBACK_SOURCE_PROBE_BUFT_LEN];
    char ids_buft[GGML_MOE_FALLBACK_SOURCE_PROBE_BUFT_LEN];
    char dst_buft[GGML_MOE_FALLBACK_SOURCE_PROBE_BUFT_LEN];
    int expert_idx;
    int src0_type;
    int src1_type;
    int ids_type;
    int dst_type;
    int64_t src0_ne[4];
    int64_t src1_ne[4];
    int64_t ids_ne[4];
    int64_t dst_ne[4];
    size_t src0_nbytes;
    size_t src1_nbytes;
    size_t ids_nbytes;
    size_t dst_nbytes;
    uint64_t rows;
    uint64_t calls;
    uint64_t fallback_us;
};

struct ggml_moe_fallback_source_probe_state {
    bool initialized;
    bool registered;
    bool enabled;
    const char * out;
    struct ggml_moe_fallback_source_probe_entry entries[GGML_MOE_FALLBACK_SOURCE_PROBE_MAX];
    int n_entries;
    uint64_t dropped;
};

static struct ggml_moe_fallback_source_probe_state ggml_moe_fallback_source_probe;


static bool ggml_moe_name_filter_matches_any(const char * filter, const char * name) {
    if (!filter || !filter[0] || !name) {
        return false;
    }

    const char * p = filter;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == ',' || *p == ':') {
            ++p;
        }
        const char * start = p;
        while (*p && *p != ',' && *p != ':') {
            ++p;
        }
        const char * end = p;
        while (end > start && (end[-1] == ' ' || end[-1] == '\t')) {
            --end;
        }
        const size_t len = (size_t) (end - start);
        if (len > 0) {
            for (const char * hit = name; *hit; ++hit) {
                if (strncmp(hit, start, len) == 0) {
                    return true;
                }
            }
        }
    }
    return false;
}

static bool ggml_moe_stream_one_name_filter_would_allow(const char * name) {
    const char * filter = getenv("GGML_MOE_STREAM_ONE_NAME_FILTER");
    if (!filter || !filter[0]) {
        return true;
    }
    return ggml_moe_name_filter_matches_any(filter, name);
}

static bool ggml_moe_stream_one_gpu_only_filter_matches(const char * name) {
    return ggml_moe_name_filter_matches_any(getenv("GGML_MOE_STREAM_ONE_GPU_ONLY_FILTER"), name);
}

static bool ggml_moe_stream_one_fast_return_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = getenv("GGML_MOE_STREAM_ONE_FAST_RETURN");
        enabled = (!env || !env[0] || env[0] != '0') ? 1 : 0;
    }
    return enabled != 0;
}

static void ggml_moe_fallback_reason_profile_report(void) {
    if (!ggml_moe_fallback_reason_profile.enabled ||
            !ggml_moe_fallback_reason_profile.out ||
            !ggml_moe_fallback_reason_profile.out[0]) {
        return;
    }

    FILE * f = fopen(ggml_moe_fallback_reason_profile.out, "w");
    if (!f) {
        fprintf(stderr,
                "[moe_fallback_reason_profile] open failed: %s\n",
                ggml_moe_fallback_reason_profile.out);
        return;
    }

    fprintf(f, "rank,role,tensor,phase,expert_idx,src0_type,expert_bytes,rows,calls,fallback_us,batch_reason,single_reason,final_reason,batch_attempts,batch_accepts,single_attempts,single_accepts\n");
    for (int i = 0; i < ggml_moe_fallback_reason_profile.n_entries; ++i) {
        const struct ggml_moe_fallback_reason_profile_entry * e =
            &ggml_moe_fallback_reason_profile.entries[i];
        fprintf(f,
                "%d,%s,%s,%s,%d,%d,%zu,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%s,%s,%s,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
                i + 1,
                e->role,
                e->tensor,
                e->phase,
                e->expert_idx,
                e->src0_type,
                e->expert_bytes,
                e->rows,
                e->calls,
                e->fallback_us,
                e->batch_reason,
                e->single_reason,
                e->final_reason,
                e->batch_attempts,
                e->batch_accepts,
                e->single_attempts,
                e->single_accepts);
    }
    fclose(f);

    fprintf(stderr,
            "[moe_fallback_reason_profile] written: %s entries=%d dropped=%" PRIu64 "\n",
            ggml_moe_fallback_reason_profile.out,
            ggml_moe_fallback_reason_profile.n_entries,
            ggml_moe_fallback_reason_profile.dropped);
}

static bool ggml_moe_fallback_reason_profile_enabled(void) {
    if (!ggml_moe_fallback_reason_profile.initialized) {
        ggml_moe_fallback_reason_profile.initialized = true;
        ggml_moe_fallback_reason_profile.out = getenv("GGML_MOE_FALLBACK_REASON_PROFILE_OUT");
        ggml_moe_fallback_reason_profile.enabled =
            ggml_moe_fallback_reason_profile.out &&
            ggml_moe_fallback_reason_profile.out[0];
        if (ggml_moe_fallback_reason_profile.enabled &&
                !ggml_moe_fallback_reason_profile.registered) {
            ggml_moe_fallback_reason_profile.registered = true;
            atexit(ggml_moe_fallback_reason_profile_report);
        }
    }

    return ggml_moe_fallback_reason_profile.enabled;
}

static void ggml_moe_fallback_reason_profile_record(
        const char * role,
        const char * tensor,
        int src0_type,
        bool prompt_phase,
        int expert_idx,
        int64_t rows,
        size_t expert_bytes,
        uint64_t fallback_us,
        const char * batch_reason,
        const char * single_reason,
        const char * final_reason,
        bool batch_attempted,
        bool batch_accepted,
        bool single_attempted,
        bool single_accepted) {
    if (!ggml_moe_fallback_reason_profile_enabled() || rows <= 0) {
        return;
    }

    const char * safe_role = role ? role : "unknown";
    const char * safe_tensor = tensor ? tensor : "<unnamed>";
    const char * safe_phase = prompt_phase ? "prompt" : "decode";
    const char * safe_batch_reason = batch_reason ? batch_reason : "unknown";
    const char * safe_single_reason = single_reason ? single_reason : "unknown";
    const char * safe_final_reason = final_reason ? final_reason : "unknown";
    int idx = -1;

    for (int i = 0; i < ggml_moe_fallback_reason_profile.n_entries; ++i) {
        struct ggml_moe_fallback_reason_profile_entry * e =
            &ggml_moe_fallback_reason_profile.entries[i];
        if (e->expert_idx == expert_idx &&
                e->src0_type == src0_type &&
                e->expert_bytes == expert_bytes &&
                strcmp(e->role, safe_role) == 0 &&
                strcmp(e->phase, safe_phase) == 0 &&
                strncmp(e->tensor, safe_tensor, GGML_MOE_FALLBACK_NAME_LEN) == 0 &&
                strncmp(e->batch_reason, safe_batch_reason, GGML_MOE_FALLBACK_REASON_LEN) == 0 &&
                strncmp(e->single_reason, safe_single_reason, GGML_MOE_FALLBACK_REASON_LEN) == 0 &&
                strncmp(e->final_reason, safe_final_reason, GGML_MOE_FALLBACK_REASON_LEN) == 0) {
            idx = i;
            break;
        }
    }

    if (idx < 0) {
        if (ggml_moe_fallback_reason_profile.n_entries >= GGML_MOE_FALLBACK_REASON_PROFILE_MAX) {
            ggml_moe_fallback_reason_profile.dropped++;
            return;
        }
        idx = ggml_moe_fallback_reason_profile.n_entries++;
        struct ggml_moe_fallback_reason_profile_entry * e =
            &ggml_moe_fallback_reason_profile.entries[idx];
        snprintf(e->tensor, GGML_MOE_FALLBACK_NAME_LEN, "%s", safe_tensor);
        snprintf(e->role, sizeof(e->role), "%s", safe_role);
        snprintf(e->phase, sizeof(e->phase), "%s", safe_phase);
        snprintf(e->batch_reason, GGML_MOE_FALLBACK_REASON_LEN, "%s", safe_batch_reason);
        snprintf(e->single_reason, GGML_MOE_FALLBACK_REASON_LEN, "%s", safe_single_reason);
        snprintf(e->final_reason, GGML_MOE_FALLBACK_REASON_LEN, "%s", safe_final_reason);
        e->expert_idx = expert_idx;
        e->src0_type = src0_type;
        e->expert_bytes = expert_bytes;
    }

    struct ggml_moe_fallback_reason_profile_entry * e =
        &ggml_moe_fallback_reason_profile.entries[idx];
    e->rows += (uint64_t) rows;
    e->calls++;
    e->fallback_us += fallback_us;
    if (batch_attempted) {
        e->batch_attempts++;
    }
    if (batch_accepted) {
        e->batch_accepts++;
    }
    if (single_attempted) {
        e->single_attempts++;
    }
    if (single_accepted) {
        e->single_accepts++;
    }
}

static void ggml_moe_fallback_source_probe_report(void);

static const char * ggml_moe_fallback_source_probe_buft_name(const struct ggml_tensor * tensor) {
    if (tensor == NULL || tensor->buffer == NULL) {
        return "none";
    }
    const char * name = ggml_backend_buffer_name(tensor->buffer);
    return name ? name : "unknown";
}

static void ggml_moe_fallback_source_probe_copy_ne(int64_t dst_ne[4], const struct ggml_tensor * tensor) {
    for (int i = 0; i < 4; ++i) {
        dst_ne[i] = tensor ? tensor->ne[i] : -1;
    }
}

static bool ggml_moe_fallback_source_probe_enabled(void) {
    if (!ggml_moe_fallback_source_probe.initialized) {
        ggml_moe_fallback_source_probe.initialized = true;
        ggml_moe_fallback_source_probe.out = getenv("GGML_MOE_FALLBACK_SOURCE_PROBE_OUT");
        ggml_moe_fallback_source_probe.enabled =
            ggml_moe_fallback_source_probe.out &&
            ggml_moe_fallback_source_probe.out[0] &&
            ggml_moe_fallback_source_probe.out[0] != '0';
        if (ggml_moe_fallback_source_probe.enabled && !ggml_moe_fallback_source_probe.registered) {
            ggml_moe_fallback_source_probe.registered = true;
            atexit(ggml_moe_fallback_source_probe_report);
        }
    }
    return ggml_moe_fallback_source_probe.enabled;
}

static void ggml_moe_fallback_source_probe_report(void) {
    if (!ggml_moe_fallback_source_probe.enabled ||
            !ggml_moe_fallback_source_probe.out ||
            !ggml_moe_fallback_source_probe.out[0]) {
        return;
    }

    FILE * f = fopen(ggml_moe_fallback_source_probe.out, "w");
    if (!f) {
        fprintf(stderr, "[moe_fallback_source_probe] open failed: %s\n", ggml_moe_fallback_source_probe.out);
        return;
    }

    fprintf(f,
            "rank,role,tensor,phase,expert_idx,rows,calls,fallback_us,batch_reason,single_reason,final_reason,"
            "src0_buft,src1_buft,ids_buft,dst_buft,src0_type,src1_type,ids_type,dst_type,"
            "src0_nbytes,src1_nbytes,ids_nbytes,dst_nbytes,"
            "src0_ne0,src0_ne1,src0_ne2,src0_ne3,src1_ne0,src1_ne1,src1_ne2,src1_ne3,"
            "ids_ne0,ids_ne1,ids_ne2,ids_ne3,dst_ne0,dst_ne1,dst_ne2,dst_ne3\n");
    for (int i = 0; i < ggml_moe_fallback_source_probe.n_entries; ++i) {
        const struct ggml_moe_fallback_source_probe_entry * e = &ggml_moe_fallback_source_probe.entries[i];
        fprintf(f,
                "%d,%s,%s,%s,%d,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%s,%s,%s,"
                "%s,%s,%s,%s,%d,%d,%d,%d,%zu,%zu,%zu,%zu,"
                "%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ","
                "%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "\n",
                i + 1,
                e->role,
                e->tensor,
                e->phase,
                e->expert_idx,
                e->rows,
                e->calls,
                e->fallback_us,
                e->batch_reason,
                e->single_reason,
                e->final_reason,
                e->src0_buft,
                e->src1_buft,
                e->ids_buft,
                e->dst_buft,
                e->src0_type,
                e->src1_type,
                e->ids_type,
                e->dst_type,
                e->src0_nbytes,
                e->src1_nbytes,
                e->ids_nbytes,
                e->dst_nbytes,
                e->src0_ne[0], e->src0_ne[1], e->src0_ne[2], e->src0_ne[3],
                e->src1_ne[0], e->src1_ne[1], e->src1_ne[2], e->src1_ne[3],
                e->ids_ne[0], e->ids_ne[1], e->ids_ne[2], e->ids_ne[3],
                e->dst_ne[0], e->dst_ne[1], e->dst_ne[2], e->dst_ne[3]);
    }
    fclose(f);

    fprintf(stderr,
            "[moe_fallback_source_probe] written: %s entries=%d dropped=%" PRIu64 "\n",
            ggml_moe_fallback_source_probe.out,
            ggml_moe_fallback_source_probe.n_entries,
            ggml_moe_fallback_source_probe.dropped);
}

static void ggml_moe_fallback_source_probe_record(
        const char * role,
        const char * tensor_name,
        bool prompt_phase,
        int expert_idx,
        int64_t rows,
        uint64_t fallback_us,
        const char * batch_reason,
        const char * single_reason,
        const char * final_reason,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        const struct ggml_tensor * ids,
        const struct ggml_tensor * dst) {
    if (!ggml_moe_fallback_source_probe_enabled() || rows <= 0) {
        return;
    }

    const char * safe_role = role ? role : "unknown";
    const char * safe_tensor = tensor_name ? tensor_name : "<unnamed>";
    const char * safe_phase = prompt_phase ? "prompt" : "decode";
    const char * safe_batch_reason = batch_reason ? batch_reason : "unknown";
    const char * safe_single_reason = single_reason ? single_reason : "unknown";
    const char * safe_final_reason = final_reason ? final_reason : "unknown";
    const char * src0_buft = ggml_moe_fallback_source_probe_buft_name(src0);
    const char * src1_buft = ggml_moe_fallback_source_probe_buft_name(src1);
    const char * ids_buft = ggml_moe_fallback_source_probe_buft_name(ids);
    const char * dst_buft = ggml_moe_fallback_source_probe_buft_name(dst);
    int idx = -1;

    for (int i = 0; i < ggml_moe_fallback_source_probe.n_entries; ++i) {
        struct ggml_moe_fallback_source_probe_entry * e = &ggml_moe_fallback_source_probe.entries[i];
        if (e->expert_idx == expert_idx &&
                strcmp(e->role, safe_role) == 0 &&
                strcmp(e->phase, safe_phase) == 0 &&
                strncmp(e->tensor, safe_tensor, GGML_MOE_FALLBACK_SOURCE_PROBE_NAME_LEN) == 0 &&
                strncmp(e->batch_reason, safe_batch_reason, GGML_MOE_FALLBACK_REASON_LEN) == 0 &&
                strncmp(e->single_reason, safe_single_reason, GGML_MOE_FALLBACK_REASON_LEN) == 0 &&
                strncmp(e->final_reason, safe_final_reason, GGML_MOE_FALLBACK_REASON_LEN) == 0 &&
                strncmp(e->src0_buft, src0_buft, GGML_MOE_FALLBACK_SOURCE_PROBE_BUFT_LEN) == 0 &&
                strncmp(e->src1_buft, src1_buft, GGML_MOE_FALLBACK_SOURCE_PROBE_BUFT_LEN) == 0 &&
                strncmp(e->ids_buft, ids_buft, GGML_MOE_FALLBACK_SOURCE_PROBE_BUFT_LEN) == 0 &&
                strncmp(e->dst_buft, dst_buft, GGML_MOE_FALLBACK_SOURCE_PROBE_BUFT_LEN) == 0) {
            idx = i;
            break;
        }
    }

    if (idx < 0) {
        if (ggml_moe_fallback_source_probe.n_entries >= GGML_MOE_FALLBACK_SOURCE_PROBE_MAX) {
            ggml_moe_fallback_source_probe.dropped++;
            return;
        }
        idx = ggml_moe_fallback_source_probe.n_entries++;
        struct ggml_moe_fallback_source_probe_entry * e = &ggml_moe_fallback_source_probe.entries[idx];
        snprintf(e->tensor, sizeof(e->tensor), "%s", safe_tensor);
        snprintf(e->role, sizeof(e->role), "%s", safe_role);
        snprintf(e->phase, sizeof(e->phase), "%s", safe_phase);
        snprintf(e->batch_reason, sizeof(e->batch_reason), "%s", safe_batch_reason);
        snprintf(e->single_reason, sizeof(e->single_reason), "%s", safe_single_reason);
        snprintf(e->final_reason, sizeof(e->final_reason), "%s", safe_final_reason);
        snprintf(e->src0_buft, sizeof(e->src0_buft), "%s", src0_buft);
        snprintf(e->src1_buft, sizeof(e->src1_buft), "%s", src1_buft);
        snprintf(e->ids_buft, sizeof(e->ids_buft), "%s", ids_buft);
        snprintf(e->dst_buft, sizeof(e->dst_buft), "%s", dst_buft);
        e->expert_idx = expert_idx;
        e->src0_type = src0 ? (int) src0->type : -1;
        e->src1_type = src1 ? (int) src1->type : -1;
        e->ids_type = ids ? (int) ids->type : -1;
        e->dst_type = dst ? (int) dst->type : -1;
        e->src0_nbytes = src0 ? ggml_nbytes(src0) : 0;
        e->src1_nbytes = src1 ? ggml_nbytes(src1) : 0;
        e->ids_nbytes = ids ? ggml_nbytes(ids) : 0;
        e->dst_nbytes = dst ? ggml_nbytes(dst) : 0;
        ggml_moe_fallback_source_probe_copy_ne(e->src0_ne, src0);
        ggml_moe_fallback_source_probe_copy_ne(e->src1_ne, src1);
        ggml_moe_fallback_source_probe_copy_ne(e->ids_ne, ids);
        ggml_moe_fallback_source_probe_copy_ne(e->dst_ne, dst);
    }

    struct ggml_moe_fallback_source_probe_entry * e = &ggml_moe_fallback_source_probe.entries[idx];
    e->rows += (uint64_t) rows;
    e->calls++;
    e->fallback_us += fallback_us;
}


// precomputed f32 table for f16 (256 KB) (simd-mappings.h)
float ggml_table_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB) (simd-mappings.h)
float ggml_table_f32_e8m0_half[1 << 8];

#if defined(__ARM_ARCH)
struct ggml_arm_arch_features_type {
    int sve_cnt;
} ggml_arm_arch_features = { 0 };
#endif

#if defined(__riscv)
struct ggml_riscv_arch_features_type {
    int rvv_vlen;
} ggml_riscv_arch_features = { 0 };
#endif

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>

#if defined(_MSC_VER) && !defined(__clang__)
#define GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_LINE))

typedef volatile LONG atomic_int;
typedef atomic_int atomic_bool;
typedef atomic_int atomic_flag;

#define ATOMIC_FLAG_INIT 0

typedef enum {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;

static void atomic_store(atomic_int * ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
static void atomic_store_explicit(atomic_int * ptr, LONG val, memory_order mo) {
    // TODO: add support for explicit memory order
    InterlockedExchange(ptr, val);
}
static LONG atomic_load(atomic_int * ptr) {
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_load_explicit(atomic_int * ptr, memory_order mo) {
    // TODO: add support for explicit memory order
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_fetch_add(atomic_int * ptr, LONG inc) {
    return InterlockedExchangeAdd(ptr, inc);
}
static LONG atomic_fetch_add_explicit(atomic_int * ptr, LONG inc, memory_order mo) {
    // TODO: add support for explicit memory order
    return InterlockedExchangeAdd(ptr, inc);
}
static atomic_bool atomic_flag_test_and_set(atomic_flag * ptr) {
    return InterlockedExchange(ptr, 1);
}
static void atomic_flag_clear(atomic_flag * ptr) {
    InterlockedExchange(ptr, 0);
}
static void atomic_thread_fence(memory_order mo) {
    MemoryBarrier();
}
#else // clang
#include <stdatomic.h>
#endif

typedef HANDLE pthread_t;

typedef DWORD thread_ret_t;
static int pthread_create(pthread_t * out, void * unused, thread_ret_t(*func)(void *), void * arg) {
    (void) unused;
    HANDLE handle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, NULL);
    if (handle == NULL)
    {
        return EAGAIN;
    }

    *out = handle;
    return 0;
}

static int pthread_join(pthread_t thread, void * unused) {
    (void) unused;
    int ret = (int) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return ret;
}

static int sched_yield (void) {
    Sleep (0);
    return 0;
}
#else

#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#if defined(__FreeBSD__)
#include <pthread_np.h>
#endif

typedef void * thread_ret_t;

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#endif

typedef pthread_t ggml_thread_t;

static bool ggml_moe_cpu_chunk_trace_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = getenv("GGML_MOE_CPU_CHUNK_TRACE_OUT");
        enabled = env && env[0] ? 1 : 0;
    }
    return enabled != 0;
}

static const char * ggml_moe_tensor_role(const char * name) {
    if (name) {
        if (strstr(name, "ffn_up_exps")) {
            return "up";
        }
        if (strstr(name, "ffn_gate_exps")) {
            return "gate";
        }
        if (strstr(name, "ffn_down_exps")) {
            return "down";
        }
    }
    return "other";
}

static bool ggml_ds4_grouped_retained_route_profile_enabled(void) {
    static int initialized = 0;
    static bool enabled = false;
    if (!initialized) {
        const char * env = getenv("GGML_DS4_GROUPED_RETAINED_ROUTE_PROFILE_OUT");
        enabled = env && env[0] && env[0] != '0';
        initialized = 1;
    }
    return enabled;
}

static const char * ggml_ds4_grouped_retained_route_profile_out(void) {
    const char * env = getenv("GGML_DS4_GROUPED_RETAINED_ROUTE_PROFILE_OUT");
    return env && env[0] && env[0] != '0' ? env : NULL;
}

static const char * ggml_ds4_grouped_retained_route_detail_out(void) {
    const char * env = getenv("GGML_DS4_GROUPED_RETAINED_ROUTE_DETAIL_OUT");
    return env && env[0] && env[0] != '0' ? env : NULL;
}

static bool ggml_ds4_grouped_retained_handoff_profile_enabled(void) {
    static int initialized = 0;
    static bool enabled = false;
    if (!initialized) {
        const char * env = getenv("GGML_DS4_GROUPED_RETAINED_HANDOFF_PROFILE_OUT");
        enabled = env && env[0] && env[0] != '0';
        initialized = 1;
    }
    return enabled;
}

static const char * ggml_ds4_grouped_retained_handoff_profile_out(void) {
    const char * env = getenv("GGML_DS4_GROUPED_RETAINED_HANDOFF_PROFILE_OUT");
    return env && env[0] && env[0] != '0' ? env : NULL;
}

static int ggml_ds4_grouped_retained_parse_layer(const char * name) {
    if (!name) {
        return -1;
    }
    const char * p = strstr(name, "blk.");
    if (p) {
        return atoi(p + 4);
    }
    const char * dash = strrchr(name, 45);
    if (dash && dash[1]) {
        return atoi(dash + 1);
    }
    return -1;
}

static bool ggml_ds4_grouped_retained_role_supported(const char * role) {
    return role &&
        (strcmp(role, "gate") == 0 ||
         strcmp(role, "up") == 0 ||
         strcmp(role, "down") == 0 ||
         strcmp(role, "up_gate") == 0);
}

static void ggml_ds4_grouped_retained_route_profile_write(
        const char * role,
        const char * tensor_name,
        int src0_type,
        const void * src0_data,
        size_t expert_stride,
        bool prompt_phase,
        const int64_t * matrix_row_counts,
        int64_t n_as,
        size_t expert_bytes,
        size_t logical_bytes_per_expert,
        bool eligible,
        const char * reason) {
    if (!ggml_ds4_grouped_retained_route_profile_enabled() ||
            !matrix_row_counts || n_as <= 0 || !tensor_name || !tensor_name[0]) {
        return;
    }

    int64_t unique_experts = 0;
    int64_t rows = 0;
    int64_t cache_contains = 0;
    int64_t cache_missing = 0;
    const bool can_query_cache =
        expert_bytes > 0 &&
        ((ggml_cuda_moe_stream_one_cache_contains && src0_data && expert_stride > 0) ||
         ggml_cuda_moe_stream_cache_contains);

    for (int64_t i = 0; i < n_as; ++i) {
        const int64_t c = matrix_row_counts[i];
        if (c <= 0) {
            continue;
        }
        ++unique_experts;
        rows += c;
        if (can_query_cache) {
            bool contains = false;
            if (ggml_cuda_moe_stream_one_cache_contains && src0_data && expert_stride > 0) {
                const char * expert_data = (const char *) src0_data + (size_t)i * expert_stride;
                contains = ggml_cuda_moe_stream_one_cache_contains(tensor_name, expert_data, expert_bytes, i);
            }
            if (!contains && ggml_cuda_moe_stream_cache_contains) {
                contains = ggml_cuda_moe_stream_cache_contains(tensor_name, expert_bytes, (int)i);
            }
            if (contains) {
                ++cache_contains;
            } else {
                ++cache_missing;
            }
        }
    }

    if (unique_experts == 0) {
        return;
    }
    if (!can_query_cache) {
        cache_contains = -1;
        cache_missing = -1;
    }

    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    static uint64_t seq = 0;
    static bool header_written = false;

    const char * path = ggml_ds4_grouped_retained_route_profile_out();
    if (!path) {
        return;
    }

    pthread_mutex_lock(&mu);
    FILE * f = fopen(path, "a");
    if (!f) {
        pthread_mutex_unlock(&mu);
        return;
    }
    if (!header_written) {
        fprintf(f,
                "seq,role,layer,phase,tensor_name,src0_type,active_experts,unique_experts,rows,"
                "expert_bytes,logical_source_bytes,cache_contains_count,cache_missing_count,"
                "eligible_grouped_retained,reason_if_ineligible\n");
        header_written = true;
    }

    const uint64_t cur_seq = ++seq;
    const size_t logical_source_bytes = (size_t)unique_experts * logical_bytes_per_expert;
    fprintf(f,
            "%" PRIu64 ",%s,%d,%s,%s,%d,%" PRId64 ",%" PRId64 ",%" PRId64
            ",%zu,%zu,%" PRId64 ",%" PRId64 ",%d,%s\n",
            cur_seq,
            role ? role : "other",
            ggml_ds4_grouped_retained_parse_layer(tensor_name),
            prompt_phase ? "prompt" : "decode",
            tensor_name,
            src0_type,
            unique_experts,
            unique_experts,
            rows,
            expert_bytes,
            logical_source_bytes,
            cache_contains,
            cache_missing,
            eligible ? 1 : 0,
            reason ? reason : "ok");
    fclose(f);

    const char * detail_path = ggml_ds4_grouped_retained_route_detail_out();
    if (detail_path && role && strcmp(role, "up_gate") != 0) {
        static bool detail_header_written = false;
        FILE * df = fopen(detail_path, "a");
        if (df) {
            if (!detail_header_written) {
                fprintf(df,
                        "seq,role,layer,phase,tensor_name,src0_type,expert_id,rows,"
                        "expert_bytes,logical_source_bytes,cache_contains,"
                        "eligible_grouped_retained,reason_if_ineligible\n");
                detail_header_written = true;
            }
            for (int64_t i = 0; i < n_as; ++i) {
                const int64_t c = matrix_row_counts[i];
                if (c <= 0) {
                    continue;
                }
                int cache_state = -1;
                if (can_query_cache) {
                    bool contains = false;
                    if (ggml_cuda_moe_stream_one_cache_contains && src0_data && expert_stride > 0) {
                        const char * expert_data = (const char *) src0_data + (size_t)i * expert_stride;
                        contains = ggml_cuda_moe_stream_one_cache_contains(tensor_name, expert_data, expert_bytes, i);
                    }
                    if (!contains && ggml_cuda_moe_stream_cache_contains) {
                        contains = ggml_cuda_moe_stream_cache_contains(tensor_name, expert_bytes, (int)i);
                    }
                    cache_state = contains ? 1 : 0;
                }
                fprintf(df,
                        "%" PRIu64 ",%s,%d,%s,%s,%d,%" PRId64 ",%" PRId64
                        ",%zu,%zu,%d,%d,%s\n",
                        cur_seq,
                        role,
                        ggml_ds4_grouped_retained_parse_layer(tensor_name),
                        prompt_phase ? "prompt" : "decode",
                        tensor_name,
                        src0_type,
                        i,
                        c,
                        expert_bytes,
                        expert_bytes,
                        cache_state,
                        eligible ? 1 : 0,
                        reason ? reason : "ok");
            }
            fclose(df);
        }
    }
    pthread_mutex_unlock(&mu);
}

static void ggml_ds4_grouped_retained_route_profile_record(
        const char * tensor_name,
        int src0_type,
        const void * src0_data,
        size_t expert_stride,
        bool prompt_phase,
        const int64_t * matrix_row_counts,
        int64_t n_as,
        size_t expert_bytes) {
    if (!ggml_ds4_grouped_retained_route_profile_enabled()) {
        return;
    }
    const char * role = ggml_moe_tensor_role(tensor_name);
    const bool supported_role = ggml_ds4_grouped_retained_role_supported(role);
    const bool supported_type = ggml_cuda_moe_stream_supports_type((enum ggml_type)src0_type);
    const bool eligible = supported_role && supported_type && expert_bytes > 0;
    const char * reason = "ok";
    if (!supported_role) {
        reason = "unsupported_role";
    } else if (!supported_type) {
        reason = "unsupported_type";
    } else if (expert_bytes == 0) {
        reason = "bad_expert_bytes";
    }
    ggml_ds4_grouped_retained_route_profile_write(
            role, tensor_name, src0_type, src0_data, expert_stride, prompt_phase,
            matrix_row_counts, n_as, expert_bytes, expert_bytes,
            eligible, reason);
}

static void ggml_ds4_grouped_retained_route_profile_record_up_gate(
        const char * up_tensor_name,
        const char * gate_tensor_name,
        int up_type,
        int gate_type,
        const void * up_data,
        size_t up_expert_stride,
        const void * gate_data,
        size_t gate_expert_stride,
        bool prompt_phase,
        const int64_t * matrix_row_counts,
        int64_t n_as,
        size_t up_expert_bytes,
        size_t gate_expert_bytes) {
    if (!ggml_ds4_grouped_retained_route_profile_enabled()) {
        return;
    }
    char combined_name[192];
    snprintf(combined_name, sizeof(combined_name), "%s+%s",
            up_tensor_name ? up_tensor_name : "up",
            gate_tensor_name ? gate_tensor_name : "gate");

    const bool supported_type =
        ggml_cuda_moe_stream_supports_type((enum ggml_type)up_type) &&
        ggml_cuda_moe_stream_supports_type((enum ggml_type)gate_type);
    const bool eligible = supported_type && up_expert_bytes > 0 && gate_expert_bytes > 0;
    const char * reason = "ok";
    if (!supported_type) {
        reason = "unsupported_type";
    } else if (up_expert_bytes == 0 || gate_expert_bytes == 0) {
        reason = "bad_expert_bytes";
    }
    (void) up_data;
    (void) up_expert_stride;
    (void) gate_data;
    (void) gate_expert_stride;
    ggml_ds4_grouped_retained_route_profile_write(
            "up_gate", combined_name, up_type, NULL, 0, prompt_phase,
            matrix_row_counts, n_as,
            up_expert_bytes > gate_expert_bytes ? up_expert_bytes : gate_expert_bytes,
            up_expert_bytes + gate_expert_bytes,
            eligible, reason);
}

struct ggml_ds4_grouped_retained_last_up_gate {
    const void * dst_data;
    int layer;
    int64_t ne01;
    int64_t dst_rows;
    int64_t active_experts;
    int64_t rows;
    char up_name[128];
    char gate_name[128];
    uint64_t serial;
};

static pthread_mutex_t ggml_ds4_grouped_retained_handoff_mu = PTHREAD_MUTEX_INITIALIZER;
static struct ggml_ds4_grouped_retained_last_up_gate ggml_ds4_grouped_retained_last_up_gate = {0};

static void ggml_ds4_grouped_retained_handoff_mark_up_gate(
        const char * up_name,
        const char * gate_name,
        const void * dst_data,
        int64_t ne01,
        const int64_t * matrix_row_counts,
        int64_t n_as,
        int64_t rows_stride) {
    if (!ggml_ds4_grouped_retained_handoff_profile_enabled() || !dst_data || !matrix_row_counts) {
        return;
    }

    int64_t active_experts = 0;
    int64_t rows = 0;
    for (int64_t e = 0; e < n_as; ++e) {
        const int64_t c = matrix_row_counts[e];
        if (c <= 0) {
            continue;
        }
        ++active_experts;
        rows += c;
    }

    pthread_mutex_lock(&ggml_ds4_grouped_retained_handoff_mu);
    ggml_ds4_grouped_retained_last_up_gate.dst_data = dst_data;
    ggml_ds4_grouped_retained_last_up_gate.layer = ggml_ds4_grouped_retained_parse_layer(up_name);
    ggml_ds4_grouped_retained_last_up_gate.ne01 = ne01;
    ggml_ds4_grouped_retained_last_up_gate.dst_rows = rows_stride;
    ggml_ds4_grouped_retained_last_up_gate.active_experts = active_experts;
    ggml_ds4_grouped_retained_last_up_gate.rows = rows;
    snprintf(ggml_ds4_grouped_retained_last_up_gate.up_name,
            sizeof(ggml_ds4_grouped_retained_last_up_gate.up_name), "%s", up_name ? up_name : "");
    snprintf(ggml_ds4_grouped_retained_last_up_gate.gate_name,
            sizeof(ggml_ds4_grouped_retained_last_up_gate.gate_name), "%s", gate_name ? gate_name : "");
    ++ggml_ds4_grouped_retained_last_up_gate.serial;
    pthread_mutex_unlock(&ggml_ds4_grouped_retained_handoff_mu);
}

static void ggml_ds4_grouped_retained_handoff_mark_glu_act(const struct ggml_tensor * dst) {
    if (!dst || !dst->data || !dst->name[0]) {
        return;
    }
    if (!strstr(dst->name, "ffn_moe_swiglu")) {
        return;
    }
    const enum ggml_glu_op op = ggml_get_glu_op(dst);
    if (op != GGML_GLU_OP_SWIGLU && op != GGML_GLU_OP_SWIGLU_OAI) {
        return;
    }
    const char * upload_env = getenv("GGML_MOE_GPU_HANDOFF_UPLOAD_GLU");
    const bool upload_enabled = upload_env && upload_env[0] && upload_env[0] != 0x30;
    if (!ggml_ds4_grouped_retained_handoff_profile_enabled() && !upload_enabled) {
        return;
    }

    const struct ggml_tensor * gate = dst->src[0];
    const struct ggml_tensor * up = dst->src[1];
    pthread_mutex_lock(&ggml_ds4_grouped_retained_handoff_mu);
    ggml_ds4_grouped_retained_last_up_gate.dst_data = dst->data;
    ggml_ds4_grouped_retained_last_up_gate.layer = ggml_ds4_grouped_retained_parse_layer(dst->name);
    ggml_ds4_grouped_retained_last_up_gate.ne01 = dst->ne[0];
    ggml_ds4_grouped_retained_last_up_gate.dst_rows = dst->ne[1] * dst->ne[2];
    ggml_ds4_grouped_retained_last_up_gate.active_experts = dst->ne[1];
    ggml_ds4_grouped_retained_last_up_gate.rows = dst->ne[1] * dst->ne[2];
    snprintf(ggml_ds4_grouped_retained_last_up_gate.up_name,
            sizeof(ggml_ds4_grouped_retained_last_up_gate.up_name), "%s",
            up && up->name[0] ? up->name : "");
    snprintf(ggml_ds4_grouped_retained_last_up_gate.gate_name,
            sizeof(ggml_ds4_grouped_retained_last_up_gate.gate_name), "%s",
            gate && gate->name[0] ? gate->name : "");
    ++ggml_ds4_grouped_retained_last_up_gate.serial;
    pthread_mutex_unlock(&ggml_ds4_grouped_retained_handoff_mu);

    if (upload_enabled && ggml_cuda_moe_stream_handoff_upload) {
        const int64_t dst_cols = dst->ne[1] * dst->ne[2];
        (void) ggml_cuda_moe_stream_handoff_upload((const float *) dst->data, dst->ne[0], dst_cols);
    }
}

static void ggml_ds4_grouped_retained_handoff_record_down(
        const char * down_name,
        const void * src1_data,
        int64_t ne00,
        int64_t ne01,
        const int64_t * matrix_row_counts,
        int64_t n_as) {
    if (!ggml_ds4_grouped_retained_handoff_profile_enabled() || !down_name || !matrix_row_counts) {
        return;
    }
    if (strcmp(ggml_moe_tensor_role(down_name), "down") != 0) {
        return;
    }

    int64_t active_experts = 0;
    int64_t rows = 0;
    for (int64_t e = 0; e < n_as; ++e) {
        const int64_t c = matrix_row_counts[e];
        if (c <= 0) {
            continue;
        }
        ++active_experts;
        rows += c;
    }
    if (active_experts == 0) {
        return;
    }

    static uint64_t seq = 0;
    static bool header_written = false;
    const char * path = ggml_ds4_grouped_retained_handoff_profile_out();
    if (!path) {
        return;
    }

    pthread_mutex_lock(&ggml_ds4_grouped_retained_handoff_mu);
    const struct ggml_ds4_grouped_retained_last_up_gate last = ggml_ds4_grouped_retained_last_up_gate;
    const uint64_t cur_seq = ++seq;
    const int down_layer = ggml_ds4_grouped_retained_parse_layer(down_name);
    const bool ptr_match = src1_data && last.dst_data == src1_data;
    const bool layer_match = last.layer == down_layer;
    const bool width_match = last.ne01 == ne00;
    const bool shape_ok = ptr_match && layer_match && width_match;

    FILE * f = fopen(path, "a");
    if (f) {
        if (!header_written) {
            fprintf(f,
                    "seq,down_layer,down_tensor,up_gate_serial,up_layer,up_tensor,gate_tensor,"
                    "ptr_match,layer_match,width_match,shape_ok,down_active_experts,down_rows,"
                    "up_active_experts,up_rows,down_ne00,down_ne01,up_ne01,up_dst_rows\n");
            header_written = true;
        }
        fprintf(f,
                "%" PRIu64 ",%d,%s,%" PRIu64 ",%d,%s,%s,%d,%d,%d,%d,%" PRId64
                ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64
                ",%" PRId64 ",%" PRId64 "\n",
                cur_seq,
                down_layer,
                down_name,
                last.serial,
                last.layer,
                last.up_name,
                last.gate_name,
                ptr_match ? 1 : 0,
                layer_match ? 1 : 0,
                width_match ? 1 : 0,
                shape_ok ? 1 : 0,
                active_experts,
                rows,
                last.active_experts,
                last.rows,
                ne00,
                ne01,
                last.ne01,
                last.dst_rows);
        fclose(f);
    }
    pthread_mutex_unlock(&ggml_ds4_grouped_retained_handoff_mu);
}

static double ggml_moe_cpu_trace_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1000.0 + (double) ts.tv_nsec / 1000000.0;
}

static FILE * ggml_moe_cpu_chunk_trace_fp(void) {
    static FILE * fp = NULL;
    static int initialized = 0;
    static pthread_mutex_t init_mu = PTHREAD_MUTEX_INITIALIZER;

    pthread_mutex_lock(&init_mu);
    if (!initialized) {
        initialized = 1;
        const char * path = getenv("GGML_MOE_CPU_CHUNK_TRACE_OUT");
        if (path && path[0]) {
            fp = fopen(path, "w");
            if (fp) {
                setvbuf(fp, NULL, _IOLBF, 0);
                fprintf(fp,
                    "seq,role,tensor,type,expert,ith,nth,cne1,ir0_start,ir0_end,ir1_start,ir1_end,src0_bytes,ms\n");
            } else {
                fprintf(stderr, "[moe_cpu_trace] failed to open trace: %s\n", path);
            }
        }
    }
    pthread_mutex_unlock(&init_mu);

    return fp;
}

static void ggml_moe_cpu_chunk_trace_write(
    const char * role,
    const char * tensor,
    enum ggml_type type,
    int expert,
    int ith,
    int nth,
    int64_t cne1,
    int64_t ir0_start,
    int64_t ir0_end,
    int64_t ir1_start,
    int64_t ir1_end,
    size_t src0_bytes,
    double ms) {
    static atomic_int seq = 0;
    static int limit = -1;

    if (limit < 0) {
        const char * env = getenv("GGML_MOE_CPU_CHUNK_TRACE_LIMIT");
        limit = env && env[0] ? atoi(env) : 200000;
        if (limit <= 0) {
            limit = 200000;
        }
    }

    const int cur_seq = atomic_fetch_add_explicit(&seq, 1, memory_order_relaxed);
    if (cur_seq >= limit) {
        return;
    }

    FILE * fp = ggml_moe_cpu_chunk_trace_fp();
    if (!fp) {
        return;
    }

    flockfile(fp);
    fprintf(fp,
        "%d,%s,%s,%d,%d,%d,%d,%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%zu,%.3f\n",
        cur_seq,
        role ? role : "",
        tensor ? tensor : "",
        (int) type,
        expert,
        ith,
        nth,
        cne1,
        ir0_start,
        ir0_end,
        ir1_start,
        ir1_end,
        src0_bytes,
        ms);
    funlockfile(fp);
}

static bool ggml_moe_cpu_fallback_touch_profile_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = getenv("GGML_MOE_CPU_FALLBACK_TOUCH_PROFILE");
        enabled = env && env[0] && env[0] != '0' ? 1 : 0;
    }
    return enabled != 0;
}

static volatile uint8_t ggml_moe_cpu_touch_sink;

static uint64_t ggml_moe_cpu_touch_pages_us(const void * ptr, size_t size) {
#if defined(__linux__)
    if (!ptr || size == 0) {
        return 0;
    }

    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        return 0;
    }

    const uintptr_t begin = (uintptr_t) ptr;
    const uintptr_t end = begin + size;
    const uintptr_t aligned_begin = begin & ~(uintptr_t) (page_size - 1);
    const uintptr_t aligned_end = (end + (uintptr_t) page_size - 1) & ~(uintptr_t) (page_size - 1);
    if (aligned_end <= aligned_begin) {
        return 0;
    }

    uint8_t acc = 0;
    const uint64_t t0 = ggml_time_us();
    for (uintptr_t p = aligned_begin; p < aligned_end; p += (uintptr_t) page_size) {
        acc ^= *(const volatile uint8_t *) p;
    }
    ggml_moe_cpu_touch_sink ^= acc;
    return ggml_time_us() - t0;
#else
    (void) ptr;
    (void) size;
    return 0;
#endif
}

static bool ggml_moe_cpu_willneed_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = getenv("GGML_MOE_CPU_WILLNEED");
        enabled = env && env[0] && env[0] != '0' ? 1 : 0;
    }
    return enabled != 0;
}

static int ggml_moe_keep_topk_updown(void) {
    static int keep_topk = -1;
    if (keep_topk < 0) {
        const char * env = getenv("GGML_MOE_KEEP_TOPK_UPDOWN");
        keep_topk = env && env[0] ? atoi(env) : 0;
        if (keep_topk < 0) {
            keep_topk = 0;
        }
    }
    return keep_topk;
}

static int ggml_moe_gpu_keep_topk_updown(void) {
    static int keep_topk = -1;
    if (keep_topk < 0) {
        const char * env = getenv("GGML_MOE_GPU_KEEP_TOPK_UPDOWN");
        keep_topk = env && env[0] ? atoi(env) : 0;
        if (keep_topk < 0) {
            keep_topk = 0;
        }
    }
    return keep_topk;
}

static bool ggml_moe_gpu_keep_topk_applies(const char * name) {
    if (!name) {
        return false;
    }
    return strstr(name, "ffn_up_exps") || strstr(name, "ffn_down_exps");
}

static bool ggml_moe_keep_topk_applies(const char * name) {
    if (!name) {
        return false;
    }
    if (strstr(name, "ffn_up_exps") || strstr(name, "ffn_down_exps")) {
        return true;
    }

    static int gate_enabled = -1;
    if (gate_enabled < 0) {
        const char * env = getenv("GGML_MOE_KEEP_TOPK_GATE");
        gate_enabled = env && env[0] && strcmp(env, "0") != 0 ? 1 : 0;
    }
    return gate_enabled && strstr(name, "ffn_gate_exps");
}

static int ggml_moe_tensor_layer(const char * name) {
    if (!name) {
        return -1;
    }
    const char * blk = strstr(name, "blk.");
    if (!blk) {
        return -1;
    }
    return atoi(blk + 4);
}

#define GGML_DS4_SPARSE_FUSED_MMVQ_MAX_LAYERS 128
#define GGML_DS4_SPARSE_FUSED_MMVQ_MAX_EXPERTS 512

struct ggml_ds4_sparse_fused_mmvq_membership_state {
    bool initialized;
    bool enabled;
    bool registered;
    bool profile_load_attempted;
    bool profile_loaded;
    const char * out;
    const char * profile;
    FILE * fp;
    pthread_mutex_t mutex;
    uint8_t hot[GGML_DS4_SPARSE_FUSED_MMVQ_MAX_LAYERS][GGML_DS4_SPARSE_FUSED_MMVQ_MAX_EXPERTS];
    uint64_t profile_pairs;
    uint64_t profile_duplicates;
    uint64_t profile_invalid;
    uint64_t records;
    uint64_t hit_records;
    uint64_t rows;
    uint64_t hit_rows;
    uint64_t prompt_rows;
    uint64_t prompt_hit_rows;
    uint64_t decode_rows;
    uint64_t decode_hit_rows;
    uint64_t source_bytes;
    uint64_t hit_source_bytes;
    uint64_t fallback_us;
    uint64_t hit_fallback_us;
    uint64_t touch_us;
    uint64_t hit_touch_us;
};

static struct ggml_ds4_sparse_fused_mmvq_membership_state ggml_ds4_sparse_fused_mmvq_membership = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
};

static void ggml_ds4_sparse_fused_mmvq_membership_add_profile_pair_locked(int layer, int expert) {
    if (layer < 0 || layer >= GGML_DS4_SPARSE_FUSED_MMVQ_MAX_LAYERS ||
            expert < 0 || expert >= GGML_DS4_SPARSE_FUSED_MMVQ_MAX_EXPERTS) {
        ggml_ds4_sparse_fused_mmvq_membership.profile_invalid++;
        return;
    }
    if (ggml_ds4_sparse_fused_mmvq_membership.hot[layer][expert]) {
        ggml_ds4_sparse_fused_mmvq_membership.profile_duplicates++;
        return;
    }
    ggml_ds4_sparse_fused_mmvq_membership.hot[layer][expert] = 1;
    ggml_ds4_sparse_fused_mmvq_membership.profile_pairs++;
}

static bool ggml_ds4_sparse_fused_mmvq_parse_json_int(const char * line, const char * key, int * out) {
    const char * p = strstr(line, key);
    if (!p) {
        return false;
    }
    p += strlen(key);
    while (*p && ((*p < '0' || *p > '9') && *p != '-')) {
        ++p;
    }
    if (!*p) {
        return false;
    }
    *out = atoi(p);
    return true;
}

static void ggml_ds4_sparse_fused_mmvq_membership_load_profile_locked(void) {
    if (ggml_ds4_sparse_fused_mmvq_membership.profile_load_attempted) {
        return;
    }
    ggml_ds4_sparse_fused_mmvq_membership.profile_load_attempted = true;

    const char * path = getenv("GGML_DS4_SPARSE_FUSED_MMVQ_PROFILE");
    if (!path || !path[0]) {
        path = getenv("GGML_DS4_SPARSE_FUSED_MMVQ_PROFILE_TSV");
    }
    if (!path || !path[0]) {
        path = getenv("DS4_SPARSE_FUSED_MMVQ_PROFILE");
    }
    ggml_ds4_sparse_fused_mmvq_membership.profile = path;
    if (!path || !path[0]) {
        fprintf(stderr, "[ds4_sparse_fused_mmvq_membership] profile env is unset\n");
        return;
    }

    FILE * f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[ds4_sparse_fused_mmvq_membership] failed to open profile: %s\n", path);
        return;
    }

    char line[2048];
    int json_layer = -1;
    int json_expert = -1;
    while (fgets(line, sizeof(line), f)) {
        int rank = 0;
        int layer = -1;
        int expert = -1;
        double hot_ms = 0.0;
        double up_hot_ms = 0.0;
        double down_hot_ms = 0.0;
        uint64_t up_calls = 0;
        uint64_t down_calls = 0;
        uint64_t fused_calls = 0;
        if (sscanf(line, "%d\t%d\t%d\t%lf\t%lf\t%lf\t%" SCNu64 "\t%" SCNu64 "\t%" SCNu64,
                    &rank, &layer, &expert, &hot_ms, &up_hot_ms, &down_hot_ms,
                    &up_calls, &down_calls, &fused_calls) >= 3) {
            ggml_ds4_sparse_fused_mmvq_membership_add_profile_pair_locked(layer, expert);
            continue;
        }

        if (strchr(line, '{')) {
            json_layer = -1;
            json_expert = -1;
        }
        (void) ggml_ds4_sparse_fused_mmvq_parse_json_int(line, "\"layer\"", &json_layer);
        (void) ggml_ds4_sparse_fused_mmvq_parse_json_int(line, "\"expert\"", &json_expert);
        if (strchr(line, '}') && json_layer >= 0 && json_expert >= 0) {
            ggml_ds4_sparse_fused_mmvq_membership_add_profile_pair_locked(json_layer, json_expert);
            json_layer = -1;
            json_expert = -1;
        }
    }
    fclose(f);

    ggml_ds4_sparse_fused_mmvq_membership.profile_loaded =
        ggml_ds4_sparse_fused_mmvq_membership.profile_pairs > 0;
    fprintf(stderr,
            "[ds4_sparse_fused_mmvq_membership] loaded profile=%s pairs=%" PRIu64
            " duplicates=%" PRIu64 " invalid=%" PRIu64 "\n",
            path,
            ggml_ds4_sparse_fused_mmvq_membership.profile_pairs,
            ggml_ds4_sparse_fused_mmvq_membership.profile_duplicates,
            ggml_ds4_sparse_fused_mmvq_membership.profile_invalid);
}

static void ggml_ds4_sparse_fused_mmvq_membership_report(void) {
    if (!ggml_ds4_sparse_fused_mmvq_membership.enabled) {
        return;
    }
    if (ggml_ds4_sparse_fused_mmvq_membership.fp) {
        fclose(ggml_ds4_sparse_fused_mmvq_membership.fp);
        ggml_ds4_sparse_fused_mmvq_membership.fp = NULL;
    }
    fprintf(stderr,
            "[ds4_sparse_fused_mmvq_membership] profile_loaded=%d pairs=%" PRIu64
            " records=%" PRIu64 " hit_records=%" PRIu64
            " rows=%" PRIu64 " hit_rows=%" PRIu64
            " prompt_rows=%" PRIu64 " prompt_hit_rows=%" PRIu64
            " decode_rows=%" PRIu64 " decode_hit_rows=%" PRIu64
            " source_bytes=%" PRIu64 " hit_source_bytes=%" PRIu64
            " fallback_us=%" PRIu64 " hit_fallback_us=%" PRIu64
            " touch_us=%" PRIu64 " hit_touch_us=%" PRIu64 "\n",
            ggml_ds4_sparse_fused_mmvq_membership.profile_loaded ? 1 : 0,
            ggml_ds4_sparse_fused_mmvq_membership.profile_pairs,
            ggml_ds4_sparse_fused_mmvq_membership.records,
            ggml_ds4_sparse_fused_mmvq_membership.hit_records,
            ggml_ds4_sparse_fused_mmvq_membership.rows,
            ggml_ds4_sparse_fused_mmvq_membership.hit_rows,
            ggml_ds4_sparse_fused_mmvq_membership.prompt_rows,
            ggml_ds4_sparse_fused_mmvq_membership.prompt_hit_rows,
            ggml_ds4_sparse_fused_mmvq_membership.decode_rows,
            ggml_ds4_sparse_fused_mmvq_membership.decode_hit_rows,
            ggml_ds4_sparse_fused_mmvq_membership.source_bytes,
            ggml_ds4_sparse_fused_mmvq_membership.hit_source_bytes,
            ggml_ds4_sparse_fused_mmvq_membership.fallback_us,
            ggml_ds4_sparse_fused_mmvq_membership.hit_fallback_us,
            ggml_ds4_sparse_fused_mmvq_membership.touch_us,
            ggml_ds4_sparse_fused_mmvq_membership.hit_touch_us);
}

static bool ggml_ds4_sparse_fused_mmvq_membership_enabled(void) {
    if (!ggml_ds4_sparse_fused_mmvq_membership.initialized) {
        ggml_ds4_sparse_fused_mmvq_membership.initialized = true;
        ggml_ds4_sparse_fused_mmvq_membership.out =
            getenv("GGML_DS4_SPARSE_FUSED_MMVQ_MEMBERSHIP_OUT");
        ggml_ds4_sparse_fused_mmvq_membership.enabled =
            ggml_ds4_sparse_fused_mmvq_membership.out &&
            ggml_ds4_sparse_fused_mmvq_membership.out[0];
        if (ggml_ds4_sparse_fused_mmvq_membership.enabled &&
                !ggml_ds4_sparse_fused_mmvq_membership.registered) {
            ggml_ds4_sparse_fused_mmvq_membership.registered = true;
            atexit(ggml_ds4_sparse_fused_mmvq_membership_report);
        }
    }
    return ggml_ds4_sparse_fused_mmvq_membership.enabled;
}

static FILE * ggml_ds4_sparse_fused_mmvq_membership_fp_locked(void) {
    if (ggml_ds4_sparse_fused_mmvq_membership.fp) {
        return ggml_ds4_sparse_fused_mmvq_membership.fp;
    }
    const char * path = ggml_ds4_sparse_fused_mmvq_membership.out;
    if (!path || !path[0]) {
        return NULL;
    }
    ggml_ds4_sparse_fused_mmvq_membership.fp = fopen(path, "w");
    if (!ggml_ds4_sparse_fused_mmvq_membership.fp) {
        fprintf(stderr, "[ds4_sparse_fused_mmvq_membership] failed to open output: %s\n", path);
        return NULL;
    }
    setvbuf(ggml_ds4_sparse_fused_mmvq_membership.fp, NULL, _IOLBF, 0);
    fprintf(ggml_ds4_sparse_fused_mmvq_membership.fp,
            "seq,phase,role,tensor,type,layer,expert,cne1,profile_hit,expert_bytes,source_bytes,"
            "fallback_us_est,touch_us,pack_mmap_source,profile_path,profile_pairs_loaded\n");
    return ggml_ds4_sparse_fused_mmvq_membership.fp;
}

static void ggml_ds4_sparse_fused_mmvq_membership_record(
        const char * name,
        int src0_type,
        bool prompt_phase,
        int expert_idx,
        int64_t count,
        size_t expert_bytes,
        uint64_t fallback_us,
        uint64_t touch_us,
        bool pack_mmap_source) {
    if (!ggml_ds4_sparse_fused_mmvq_membership_enabled() || count <= 0) {
        return;
    }

    const char * role = ggml_moe_tensor_role(name);
    if (strcmp(role, "up") != 0 && strcmp(role, "down") != 0) {
        return;
    }
    const int layer = ggml_moe_tensor_layer(name);
    if (layer < 0 || expert_idx < 0) {
        return;
    }

    pthread_mutex_lock(&ggml_ds4_sparse_fused_mmvq_membership.mutex);
    ggml_ds4_sparse_fused_mmvq_membership_load_profile_locked();
    const bool profile_hit =
        layer < GGML_DS4_SPARSE_FUSED_MMVQ_MAX_LAYERS &&
        expert_idx < GGML_DS4_SPARSE_FUSED_MMVQ_MAX_EXPERTS &&
        ggml_ds4_sparse_fused_mmvq_membership.hot[layer][expert_idx] != 0;

    const uint64_t rows = (uint64_t) count;
    const uint64_t source_bytes = (uint64_t) expert_bytes;

    const uint64_t seq = ggml_ds4_sparse_fused_mmvq_membership.records++;
    ggml_ds4_sparse_fused_mmvq_membership.rows += rows;
    ggml_ds4_sparse_fused_mmvq_membership.source_bytes += source_bytes;
    ggml_ds4_sparse_fused_mmvq_membership.fallback_us += fallback_us;
    ggml_ds4_sparse_fused_mmvq_membership.touch_us += touch_us;
    if (prompt_phase) {
        ggml_ds4_sparse_fused_mmvq_membership.prompt_rows += rows;
    } else {
        ggml_ds4_sparse_fused_mmvq_membership.decode_rows += rows;
    }
    if (profile_hit) {
        ggml_ds4_sparse_fused_mmvq_membership.hit_records++;
        ggml_ds4_sparse_fused_mmvq_membership.hit_rows += rows;
        ggml_ds4_sparse_fused_mmvq_membership.hit_source_bytes += source_bytes;
        ggml_ds4_sparse_fused_mmvq_membership.hit_fallback_us += fallback_us;
        ggml_ds4_sparse_fused_mmvq_membership.hit_touch_us += touch_us;
        if (prompt_phase) {
            ggml_ds4_sparse_fused_mmvq_membership.prompt_hit_rows += rows;
        } else {
            ggml_ds4_sparse_fused_mmvq_membership.decode_hit_rows += rows;
        }
    }

    FILE * fp = ggml_ds4_sparse_fused_mmvq_membership_fp_locked();
    if (fp) {
        fprintf(fp,
                "%" PRIu64 ",%s,%s,%s,%d,%d,%d,%" PRId64 ",%d,%zu,%" PRIu64
                ",%" PRIu64 ",%" PRIu64 ",%d,%s,%" PRIu64 "\n",
                seq,
                prompt_phase ? "prompt" : "decode",
                role,
                name ? name : "",
                src0_type,
                layer,
                expert_idx,
                count,
                profile_hit ? 1 : 0,
                expert_bytes,
                source_bytes,
                fallback_us,
                touch_us,
                pack_mmap_source ? 1 : 0,
                ggml_ds4_sparse_fused_mmvq_membership.profile ?
                    ggml_ds4_sparse_fused_mmvq_membership.profile : "",
                ggml_ds4_sparse_fused_mmvq_membership.profile_pairs);
    }
    pthread_mutex_unlock(&ggml_ds4_sparse_fused_mmvq_membership.mutex);
}

enum { GGML_MOE_KEEP_TOPK_MAX_SCHEDULE = 16 };

struct ggml_moe_keep_topk_schedule_entry {
    int start;
    int end;
    int keep_topk;
};

struct ggml_moe_keep_topk_schedule_state {
    int initialized;
    int count;
    struct ggml_moe_keep_topk_schedule_entry entries[GGML_MOE_KEEP_TOPK_MAX_SCHEDULE];
};

static void ggml_moe_keep_topk_schedule_init(
        struct ggml_moe_keep_topk_schedule_state * state,
        const char * env_name) {
    if (state->initialized) {
        return;
    }
    state->initialized = 1;
    const char * env = getenv(env_name);
    if (!env || !env[0]) {
        return;
    }

    const char * p = env;
    while (*p && state->count < GGML_MOE_KEEP_TOPK_MAX_SCHEDULE) {
        while (*p == ' ' || *p == '\t' || *p == ',') {
            ++p;
        }
        int start = -1;
        int end = -1;
        int keep = 0;
        int consumed = 0;
        if (sscanf(p, "%d-%d:%d%n", &start, &end, &keep, &consumed) == 3 ||
            sscanf(p, "%d:%d%n", &start, &keep, &consumed) == 2) {
            if (end < 0) {
                end = start;
            }
            if (start > end) {
                const int tmp = start;
                start = end;
                end = tmp;
            }
            if (keep < 0) {
                keep = 0;
            }
            state->entries[state->count++] = (struct ggml_moe_keep_topk_schedule_entry) {
                start,
                end,
                keep,
            };
            p += consumed;
            while (*p && *p != ',') {
                ++p;
            }
            continue;
        }
        break;
    }
}

static int ggml_moe_keep_topk_schedule_lookup(
        struct ggml_moe_keep_topk_schedule_state * state,
        const char * env_name,
        int layer) {
    ggml_moe_keep_topk_schedule_init(state, env_name);
    for (int i = 0; i < state->count; ++i) {
        if (layer >= state->entries[i].start && layer <= state->entries[i].end) {
            return state->entries[i].keep_topk;
        }
    }
    return -1;
}

static int ggml_moe_keep_topk_for_tensor(const char * name) {
    if (!ggml_moe_keep_topk_applies(name)) {
        return 0;
    }

    const int fallback_keep_topk = ggml_moe_keep_topk_updown();
    const int layer = ggml_moe_tensor_layer(name);

    const char * role = ggml_moe_tensor_role(name);
    static struct ggml_moe_keep_topk_schedule_state up_schedule;
    static struct ggml_moe_keep_topk_schedule_state gate_schedule;
    static struct ggml_moe_keep_topk_schedule_state down_schedule;
    if (strcmp(role, "up") == 0) {
        const int keep = ggml_moe_keep_topk_schedule_lookup(
                &up_schedule, "GGML_MOE_KEEP_TOPK_UP_LAYER_SCHEDULE", layer);
        if (keep >= 0) {
            return keep;
        }
    } else if (strcmp(role, "gate") == 0) {
        const int keep = ggml_moe_keep_topk_schedule_lookup(
                &gate_schedule, "GGML_MOE_KEEP_TOPK_GATE_LAYER_SCHEDULE", layer);
        if (keep >= 0) {
            return keep;
        }
    } else if (strcmp(role, "down") == 0) {
        const int keep = ggml_moe_keep_topk_schedule_lookup(
                &down_schedule, "GGML_MOE_KEEP_TOPK_DOWN_LAYER_SCHEDULE", layer);
        if (keep >= 0) {
            return keep;
        }
    }

    static struct ggml_moe_keep_topk_schedule_state schedule;
    const int scheduled_keep = ggml_moe_keep_topk_schedule_lookup(
            &schedule, "GGML_MOE_KEEP_TOPK_LAYER_SCHEDULE", layer);
    if (scheduled_keep >= 0) {
        return scheduled_keep;
    }

    static int initialized = 0;
    static int layer_start = -1;
    static int layer_end = -1;
    static int layer_keep_topk = 0;
    if (!initialized) {
        initialized = 1;
        const char * range = getenv("GGML_MOE_KEEP_TOPK_LAYER_RANGE");
        const char * value = getenv("GGML_MOE_KEEP_TOPK_LAYER_VALUE");
        if (range && range[0] && value && value[0]) {
            int start = -1;
            int end = -1;
            if (sscanf(range, "%d-%d", &start, &end) == 2) {
                if (start > end) {
                    const int tmp = start;
                    start = end;
                    end = tmp;
                }
                layer_start = start;
                layer_end = end;
                layer_keep_topk = atoi(value);
                if (layer_keep_topk < 0) {
                    layer_keep_topk = 0;
                }
            }
        }
    }

    if (layer_keep_topk > 0 && layer >= layer_start && layer <= layer_end) {
        return layer_keep_topk;
    }

    return fallback_keep_topk;
}

static void ggml_moe_cpu_willneed_pages(const void * ptr, size_t size) {
#if defined(__linux__)
    if (!ptr || size == 0) {
        return;
    }

    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        return;
    }

    const uintptr_t begin = (uintptr_t) ptr;
    const uintptr_t end = begin + size;
    const uintptr_t aligned_begin = begin & ~(uintptr_t) (page_size - 1);
    const uintptr_t aligned_end = (end + (uintptr_t) page_size - 1) & ~(uintptr_t) (page_size - 1);
    if (aligned_end > aligned_begin) {
        (void) madvise((void *) aligned_begin, aligned_end - aligned_begin, MADV_WILLNEED);
    }
#else
    (void) ptr;
    (void) size;
#endif
}

static bool ggml_moe_stream_compare_cpu_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = getenv("GGML_MOE_STREAM_COMPARE_CPU_OUT");
        enabled = env && env[0] ? 1 : 0;
    }
    return enabled != 0;
}

static bool ggml_moe_stream_q80_probe_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = getenv("GGML_MOE_STREAM_Q80_PROBE_OUT");
        enabled = env && env[0] ? 1 : 0;
    }
    return enabled != 0;
}

static bool ggml_moe_stream_q80_write_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = getenv("GGML_MOE_STREAM_Q80_WRITE_NAME_FILTER");
        enabled = env && env[0] ? 1 : 0;
    }
    return enabled != 0;
}

static bool ggml_moe_stream_q80_skip_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = getenv("GGML_MOE_STREAM_Q80_SKIP_NAME_FILTER");
        enabled = env && env[0] ? 1 : 0;
    }
    return enabled != 0;
}

static bool ggml_moe_stream_q80_hot_batch_probe_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = getenv("GGML_MOE_STREAM_Q80_HOT_BATCH_PROBE_OUT");
        enabled = env && env[0] ? 1 : 0;
    }
    return enabled != 0;
}

static bool ggml_moe_cpu_batch_microprobe_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = getenv("GGML_MOE_CPU_BATCH_MICROPROBE_OUT");
        enabled = env && env[0] ? 1 : 0;
    }
    return enabled != 0;
}

static bool ggml_moe_cpu_batch_microprobe_name_allows(const char * name) {
    const char * filter = getenv("GGML_MOE_CPU_BATCH_MICROPROBE_NAME_FILTER");
    return !filter || !filter[0] || (name && strstr(name, filter));
}

static int ggml_moe_cpu_batch_microprobe_env_int(const char * name, int fallback) {
    const char * env = getenv(name);
    if (!env || !env[0]) {
        return fallback;
    }
    return atoi(env);
}

static FILE * ggml_moe_cpu_batch_microprobe_fp(void) {
    static FILE * fp = NULL;
    static int initialized = 0;
    static pthread_mutex_t init_mu = PTHREAD_MUTEX_INITIALIZER;

    pthread_mutex_lock(&init_mu);
    if (!initialized) {
        initialized = 1;
        const char * path = getenv("GGML_MOE_CPU_BATCH_MICROPROBE_OUT");
        if (path && path[0]) {
            fp = fopen(path, "w");
            if (fp) {
                setvbuf(fp, NULL, _IOLBF, 0);
                fprintf(fp,
                    "seq,role,tensor,type,expert,nth,repeats,cne1,probe_cols,probe_rows,dot_calls,diff_count,src0_bytes,q80_bytes,total_bytes,wall_us,src0_gib_s,total_gib_s,us_per_dot,max_abs,mean_abs,sink\n");
            } else {
                fprintf(stderr, "[moe_cpu_batch_microprobe] failed to open report: %s\n", path);
            }
        }
    }
    pthread_mutex_unlock(&init_mu);

    return fp;
}

static void ggml_moe_cpu_batch_microprobe_write(
        int seq,
        const char * role,
        const char * tensor,
        enum ggml_type type,
        int expert,
        int nth,
        int repeats,
        int64_t cne1,
        int64_t probe_cols,
        int64_t probe_rows,
        uint64_t dot_calls,
        uint64_t diff_count,
        uint64_t src0_bytes,
        uint64_t q80_bytes,
        uint64_t wall_us,
        double max_abs,
        double sum_abs,
        double sink) {
    FILE * fp = ggml_moe_cpu_batch_microprobe_fp();
    if (!fp || wall_us == 0) {
        return;
    }

    const uint64_t total_bytes = src0_bytes + q80_bytes;
    const double sec = (double) wall_us / 1000000.0;
    const double gib = 1024.0 * 1024.0 * 1024.0;
    const double src0_gib_s = sec > 0.0 ? ((double) src0_bytes / gib) / sec : 0.0;
    const double total_gib_s = sec > 0.0 ? ((double) total_bytes / gib) / sec : 0.0;
    const double us_per_dot = dot_calls > 0 ? (double) wall_us / (double) dot_calls : 0.0;
    const double mean_abs = diff_count > 0 ? sum_abs / (double) diff_count : 0.0;

    flockfile(fp);
    fprintf(fp,
        "%d,%s,%s,%d,%d,%d,%d,%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%.3f,%.3f,%.6f,%.9g,%.9g,%.9g\n",
        seq,
        role ? role : "",
        tensor ? tensor : "",
        (int) type,
        expert,
        nth,
        repeats,
        cne1,
        probe_cols,
        probe_rows,
        dot_calls,
        diff_count,
        src0_bytes,
        q80_bytes,
        total_bytes,
        wall_us,
        src0_gib_s,
        total_gib_s,
        us_per_dot,
        max_abs,
        mean_abs,
        sink);
    funlockfile(fp);
}

static FILE * ggml_moe_stream_compare_cpu_fp(void) {
    static FILE * fp = NULL;
    static int initialized = 0;
    static pthread_mutex_t init_mu = PTHREAD_MUTEX_INITIALIZER;

    pthread_mutex_lock(&init_mu);
    if (!initialized) {
        initialized = 1;
        const char * path = getenv("GGML_MOE_STREAM_COMPARE_CPU_OUT");
        if (path && path[0]) {
            fp = fopen(path, "w");
            if (fp) {
                setvbuf(fp, NULL, _IOLBF, 0);
                fprintf(fp,
                    "seq,tensor,type,expert,cne1,count,max_abs,mean_abs,max_rel,max_k,max_row,max_col,cpu_value,gpu_value\n");
            } else {
                fprintf(stderr, "[moe_stream_compare] failed to open compare trace: %s\n", path);
            }
        }
    }
    pthread_mutex_unlock(&init_mu);

    return fp;
}

static bool ggml_moe_stream_compare_cpu_take_record(int * out_seq) {
    static atomic_int seq = 0;
    static int limit = -1;
    if (limit < 0) {
        const char * env = getenv("GGML_MOE_STREAM_COMPARE_CPU_LIMIT");
        limit = env && env[0] ? atoi(env) : 32;
        if (limit <= 0) {
            limit = 32;
        }
    }

    const int cur_seq = atomic_fetch_add_explicit(&seq, 1, memory_order_relaxed);
    if (cur_seq >= limit) {
        return false;
    }
    *out_seq = cur_seq;
    return true;
}

static void ggml_moe_stream_compare_cpu_write(
    int seq,
    const char * tensor,
    enum ggml_type type,
    int expert,
    int64_t cne1,
    int64_t count,
    double max_abs,
    double mean_abs,
    double max_rel,
    int64_t max_k,
    int32_t max_row,
    int32_t max_col,
    float cpu_value,
    float gpu_value) {
    FILE * fp = ggml_moe_stream_compare_cpu_fp();
    if (!fp) {
        return;
    }

    flockfile(fp);
    fprintf(fp,
        "%d,%s,%d,%d,%" PRId64 ",%" PRId64 ",%.9g,%.9g,%.9g,%" PRId64 ",%d,%d,%.9g,%.9g\n",
        seq,
        tensor ? tensor : "",
        (int) type,
        expert,
        cne1,
        count,
        max_abs,
        mean_abs,
        max_rel,
        max_k,
        max_row,
        max_col,
        (double) cpu_value,
        (double) gpu_value);
    funlockfile(fp);
}

static FILE * ggml_moe_stream_compare_block_fp(void) {
    static FILE * fp = NULL;
    static int initialized = 0;
    static pthread_mutex_t init_mu = PTHREAD_MUTEX_INITIALIZER;

    pthread_mutex_lock(&init_mu);
    if (!initialized) {
        initialized = 1;
        const char * path = getenv("GGML_MOE_STREAM_COMPARE_BLOCK_OUT");
        if (path && path[0]) {
            fp = fopen(path, "w");
            if (fp) {
                setvbuf(fp, NULL, _IOLBF, 0);
                fprintf(fp,
                    "seq,tensor,type,expert,max_k,row_id,i12,out_col,block,cpu_wdata_q80_part,post_src1_q80_part,abs_part_diff,d_wdata,d_post,e,amax_post,sumi_wdata,sumi_post,cpu_wdata_total,post_src1_total,gpu_value\n");
            } else {
                fprintf(stderr, "[moe_stream_compare] failed to open block trace: %s\n", path);
            }
        }
    }
    pthread_mutex_unlock(&init_mu);

    return fp;
}

static const int8_t ggml_moe_stream_mxfp4_values[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

static void ggml_moe_stream_compare_block_write(
    int seq,
    const char * tensor,
    enum ggml_type type,
    int expert,
    int64_t max_k,
    int32_t row_id,
    int64_t i12,
    int32_t out_col,
    int64_t block,
    double cpu_wdata_q80_part,
    double post_src1_q80_part,
    double d_wdata,
    double d_post,
    int e,
    double amax_post,
    int sumi_wdata,
    int sumi_post,
    double cpu_wdata_total,
    double post_src1_total,
    float gpu_value) {
    FILE * fp = ggml_moe_stream_compare_block_fp();
    if (!fp) {
        return;
    }

    flockfile(fp);
    fprintf(fp,
        "%d,%s,%d,%d,%" PRId64 ",%d,%" PRId64 ",%d,%" PRId64 ",%.9g,%.9g,%.9g,%.9g,%.9g,%d,%.9g,%d,%d,%.9g,%.9g,%.9g\n",
        seq,
        tensor ? tensor : "",
        (int) type,
        expert,
        max_k,
        row_id,
        i12,
        out_col,
        block,
        cpu_wdata_q80_part,
        post_src1_q80_part,
        fabs(cpu_wdata_q80_part - post_src1_q80_part),
        d_wdata,
        d_post,
        e,
        amax_post,
        sumi_wdata,
        sumi_post,
        cpu_wdata_total,
        post_src1_total,
        (double) gpu_value);
    funlockfile(fp);
}

#define GGML_THREADPOOL_N_THREADS_MASK (0xffffU)
#define GGML_THREADPOOL_N_THREADS_BITS (16)

#if defined(__APPLE__)
#include <unistd.h>
#include <mach/mach.h>
#include <TargetConditionals.h>
#endif

static const struct ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_fp32,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f32,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [GGML_TYPE_F16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_fp16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f16,
        .vec_dot_type             = GGML_TYPE_F16,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q1_0] = {
        .from_float               = quantize_row_q1_0,
        .vec_dot                  = ggml_vec_dot_q1_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_0] = {
        .from_float               = quantize_row_q4_0,
        .vec_dot                  = ggml_vec_dot_q4_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q4_1] = {
        .from_float               = quantize_row_q4_1,
        .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q5_0] = {
        .from_float               = quantize_row_q5_0,
        .vec_dot                  = ggml_vec_dot_q5_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q5_1] = {
        .from_float               = quantize_row_q5_1,
        .vec_dot                  = ggml_vec_dot_q5_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_0] = {
        .from_float               = quantize_row_q8_0,
        .vec_dot                  = ggml_vec_dot_q8_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q8_1] = {
        .from_float               = quantize_row_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_MXFP4] = {
        .from_float               = quantize_row_mxfp4,
        .vec_dot                  = ggml_vec_dot_mxfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_NVFP4] = {
        .from_float               = quantize_row_nvfp4,
        .vec_dot                  = ggml_vec_dot_nvfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_F8_E4M3_B128] = {
        .from_float               = quantize_row_f8_e4m3_b128,
        .vec_dot                  = ggml_vec_dot_f8_e4m3_b128_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_K] = {
        .from_float               = quantize_row_q2_K,
        .vec_dot                  = ggml_vec_dot_q2_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q3_K] = {
        .from_float               = quantize_row_q3_K,
        .vec_dot                  = ggml_vec_dot_q3_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_K] = {
        .from_float               = quantize_row_q4_K,
        .vec_dot                  = ggml_vec_dot_q4_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q5_K] = {
        .from_float               = quantize_row_q5_K,
        .vec_dot                  = ggml_vec_dot_q5_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q6_K] = {
        .from_float               = quantize_row_q6_K,
        .vec_dot                  = ggml_vec_dot_q6_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_IQ2_XXS] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XS] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq2_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_XXS] = {
        // NOTE: from_float for iq3 and iq2_s was removed because these quants require initialization in ggml_quantize_init
        //.from_float               = quantize_row_iq3_xxs,
        .vec_dot                  = ggml_vec_dot_iq3_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_S] = {
        //.from_float               = quantize_row_iq3_s,
        .vec_dot                  = ggml_vec_dot_iq3_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_S] = {
        //.from_float               = quantize_row_iq2_s,
        .vec_dot                  = ggml_vec_dot_iq2_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_S] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_M] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq1_m_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_NL] = {
        .from_float               = quantize_row_iq4_nl,
        .vec_dot                  = ggml_vec_dot_iq4_nl_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_XS] = {
        .from_float               = quantize_row_iq4_xs,
        .vec_dot                  = ggml_vec_dot_iq4_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_K] = {
        .from_float               = quantize_row_q8_K,
    },
    [GGML_TYPE_BF16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_bf16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_bf16,
        .vec_dot_type             = GGML_TYPE_BF16,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ1_0] = {
        .from_float               = quantize_row_tq1_0,
        .vec_dot                  = ggml_vec_dot_tq1_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ2_0] = {
        .from_float               = quantize_row_tq2_0,
        .vec_dot                  = ggml_vec_dot_tq2_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_I32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_i32,
    },
};

const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type) {
    return &type_traits_cpu[type];
}

//
// Threading defs
//

typedef pthread_t          ggml_thread_t;

#if defined(_WIN32)

typedef CONDITION_VARIABLE ggml_cond_t;
typedef SRWLOCK            ggml_mutex_t;

#define ggml_mutex_init(m)   InitializeSRWLock(m)
#define ggml_mutex_destroy(m)
#define ggml_mutex_lock(m)   AcquireSRWLockExclusive(m)
#define ggml_mutex_unlock(m) ReleaseSRWLockExclusive(m)
#define ggml_mutex_lock_shared(m)   AcquireSRWLockShared(m)
#define ggml_mutex_unlock_shared(m) ReleaseSRWLockShared(m)

#define ggml_cond_init(c)    InitializeConditionVariable(c)
#define ggml_cond_destroy(c)
#define ggml_cond_wait(c, m) SleepConditionVariableSRW(c, m, INFINITE, CONDITION_VARIABLE_LOCKMODE_SHARED)
#define ggml_cond_broadcast(c) WakeAllConditionVariable(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#else

typedef pthread_cond_t     ggml_cond_t;
typedef pthread_mutex_t    ggml_mutex_t;

#define ggml_mutex_init(m)          pthread_mutex_init(m, NULL)
#define ggml_mutex_destroy(m)       pthread_mutex_destroy(m)
#define ggml_mutex_lock(m)          pthread_mutex_lock(m)
#define ggml_mutex_unlock(m)        pthread_mutex_unlock(m)
#define ggml_mutex_lock_shared(m)   pthread_mutex_lock(m)
#define ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#define ggml_lock_init(x)    UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#if defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
#define ggml_lock_lock(x)    _mm_pause()
#else
#define ggml_lock_lock(x)    UNUSED(x)
#endif
#define ggml_lock_unlock(x)  UNUSED(x)

#define GGML_LOCK_INITIALIZER 0
#define ggml_cond_init(c)      pthread_cond_init(c, NULL)
#define ggml_cond_destroy(c)   pthread_cond_destroy(c)
#define ggml_cond_wait(c, m)   pthread_cond_wait(c, m)
#define ggml_cond_broadcast(c) pthread_cond_broadcast(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#endif

// Threadpool def
struct ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    struct ggml_cgraph * cgraph;
    struct ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_graph;       // updated when there is work to be done (i.e each graph) holds graph and active thread counts.
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int  abort;        // Used for aborting processing of a graph

    struct ggml_compute_state * workers;   // per thread state
    int          n_threads;   // Number of threads in the pool
    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    enum ggml_status ec;
};

// Per-thread state
struct ggml_compute_state {
#ifndef GGML_USE_OPENMP
    ggml_thread_t thrd;
    int  last_graph;
    bool pending;
#endif
    bool cpumask[GGML_MAX_N_THREADS];
    struct ggml_threadpool * threadpool;
    int ith;
};

// Helpers for polling loops
#if defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
static inline void ggml_thread_cpu_relax(void) {
    __asm__ volatile("yield" ::: "memory");
}
#elif defined(__x86_64__)
static inline void ggml_thread_cpu_relax(void) {
    _mm_pause();
}
#elif defined(__riscv)
static inline void ggml_thread_cpu_relax(void) {
    #ifdef __riscv_zihintpause
        __asm__ __volatile__ ("pause");
    #else
        /* Encoding of the pause instruction */
        __asm__ __volatile__ (".4byte 0x100000F");
    #endif
}
#else
static inline void ggml_thread_cpu_relax(void) {;}
#endif

//
// NUMA support
//

#define GGML_NUMA_MAX_NODES 8
#define GGML_NUMA_MAX_CPUS 512

struct ggml_numa_node {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

struct ggml_numa_nodes {
    enum ggml_numa_strategy numa_strategy;
    struct ggml_numa_node nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t total_cpus; // hardware threads on system
    uint32_t current_node; // node on which main process is execting
#if defined(__gnu_linux__)
    cpu_set_t cpuset; // cpuset from numactl
#else
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#endif
};

//
// ggml state
//

struct ggml_state {
    struct ggml_numa_nodes numa;
};

static struct ggml_state g_state = {0};

void ggml_barrier(struct ggml_threadpool * tp) {
    int n_threads = atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK;
    if (n_threads == 1) {
        return;
    }

#ifdef GGML_USE_OPENMP
    #pragma omp barrier
#else
    int n_passed = atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    int n_barrier = atomic_fetch_add_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    if (n_barrier == (n_threads - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (full seq-cst fence)
        atomic_fetch_add_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
        return;
    }

    // wait for other threads
    while (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
        ggml_thread_cpu_relax();
    }

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
#endif
}

void ggml_threadpool_chunk_set(struct ggml_threadpool * tp, int value) {
    atomic_store_explicit(&tp->current_chunk, value, memory_order_relaxed);
}

int ggml_threadpool_chunk_add(struct ggml_threadpool * tp, int value) {
    return atomic_fetch_add_explicit(&tp->current_chunk, value, memory_order_relaxed);
}

#if defined(__gnu_linux__)
static cpu_set_t ggml_get_numa_affinity(void) {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    return cpuset;
}
#else
static uint32_t ggml_get_numa_affinity(void) {
    return 0; // no NUMA support
}
#endif

void ggml_numa_init(enum ggml_numa_strategy numa_flag) {
    if (g_state.numa.n_nodes > 0) {
        fprintf(stderr, "ggml_numa_init: NUMA already initialized\n");

        return;
    }

#if defined(__gnu_linux__)
    struct stat st;
    char path[256];
    int rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    GGML_PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = ggml_get_numa_affinity();

    // enumerate nodes
    while (g_state.numa.n_nodes < GGML_NUMA_MAX_NODES) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
        GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.n_nodes;
    }

    // enumerate CPUs
    while (g_state.numa.total_cpus < GGML_NUMA_MAX_CPUS) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u", g_state.numa.total_cpus);
        GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.total_cpus;
    }

    GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.total_cpus);

    // figure out which node we're on
    uint current_cpu;
    int getcpu_ret = 0;
#if __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_node);
#else
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   if !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_node);
#endif

    if (g_state.numa.n_nodes < 1 || g_state.numa.total_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        return;
    }

    GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_node, current_cpu);

    for (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
        struct ggml_numa_node * node = &g_state.numa.nodes[n];
        GGML_PRINT_DEBUG("CPUs on node %u:", n);
        node->n_cpus = 0;
        for (uint32_t c = 0; c < g_state.numa.total_cpus; ++c) {
            rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u/cpu%u", n, c);
            GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
            if (stat(path, &st) == 0) {
                node->cpus[node->n_cpus++] = c;
                GGML_PRINT_DEBUG(" %u", c);
            }
        }
        GGML_PRINT_DEBUG("\n");
    }

    if (ggml_is_numa()) {
        FILE *fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
        if (fptr != NULL) {
            char buf[42];
            if (fgets(buf, sizeof(buf), fptr) && strncmp(buf, "0\n", sizeof(buf)) != 0) {
                GGML_LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            fclose(fptr);
        }
    }
#else
    UNUSED(numa_flag);
    // TODO
#endif
}

bool ggml_is_numa(void) {
    return g_state.numa.n_nodes > 1;
}

#if defined(__ARM_ARCH)
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
static void ggml_init_arm_arch_features(void) {
    ggml_arm_arch_features.sve_cnt = svcntb();
}
#else
static void ggml_init_arm_arch_features(void) {}
#endif
#endif // __ARM_ARCH

#if defined(__riscv) && defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
static void ggml_init_riscv_arch_features(void) {
    ggml_riscv_arch_features.rvv_vlen = __riscv_vlenb();
}
#else
static void ggml_init_riscv_arch_features(void) {}
#endif

struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    ggml_set_i32(result, value);

    return result;
}

struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_f32(result, value);

    return result;
}

struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(value));
                }
            } break;
        case GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(value));
                }
            } break;
        case GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }

    return tensor;
}

struct ggml_tensor * ggml_set_f32(struct ggml_tensor * tensor, float value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(value));
                }
            } break;
        case GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(ggml_bf16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(value));
                }
            } break;
        case GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }

    return tensor;
}

int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                return ((int8_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                return ((int16_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                return ((int32_t *)(tensor->data))[i];
            }
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_bf16_t));
                return GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(float));
                return ((float *)(tensor->data))[i];
            }
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_i32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_bf16_t));
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(float));
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case GGML_TYPE_F16:
            return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        case GGML_TYPE_BF16:
            return GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        case GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return ggml_get_f32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                return ((int8_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I16:
            {
                return ((int16_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I32:
            {
                return ((int32_t *)(tensor->data))[i];
            }
        case GGML_TYPE_F16:
            {
                return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_BF16:
            {
                return GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_F32:
            {
                return ((float *)(tensor->data))[i];
            }
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_f32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

float ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case GGML_TYPE_F16:
            return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        case GGML_TYPE_BF16:
            return GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        case GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

////////////////////////////////////////////////////////////////////////////////

// ggml_compute_forward_mul_mat

static void ggml_compute_forward_mul_mat_one_chunk(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst,
    const enum ggml_type type,
    const int64_t num_rows_per_vec_dot,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t const vec_dot      = type_traits_cpu[type].vec_dot;
    enum ggml_type const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);

    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    // block-tiling attempt
    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    float tmp[32];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                const int64_t i13 = (ir1 / (ne12 * ne1));
                const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                const int64_t i03 = i13 / r3;
                const int64_t i02 = i12 / r2;

                const int64_t i1 = i11;
                const int64_t i2 = i12;
                const int64_t i3 = i13;

                const char * src0_row = (const char*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                float * dst_col = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (num_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                }

                for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
                }
            }
        }
    }
}

void ggml_compute_forward_mul_mat(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    enum ggml_type           const vec_dot_type         = type_traits_cpu[src0->type].vec_dot_type;
    ggml_from_float_t        const from_float           = type_traits_cpu[vec_dot_type].from_float;
    int64_t                  const vec_dot_num_rows     = type_traits_cpu[src0->type].nrows;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    const char * dense_min_env = getenv("GGML_KIMI_DENSE_STREAM_MIN_TOKENS");
    const int dense_min_tokens = dense_min_env ? atoi(dense_min_env) : 1;
    if (ne11 >= (dense_min_tokens > 0 ? dense_min_tokens : 1) && ggml_cuda_kimi_dense_stream_mul_mat &&
            ggml_kimi_dense_stream_layer_selected(src0->name) &&
            (!ggml_cuda_kimi_dense_stream_should_intercept ||
                ggml_cuda_kimi_dense_stream_should_intercept(src0->name)) &&
            src1->type == GGML_TYPE_F32 &&
            dst->type == GGML_TYPE_F32) {
        if (ith == 0) {
            const bool done = ggml_cuda_kimi_dense_stream_mul_mat(
                src0->type,
                src0->name,
                src0->data,
                ggml_nbytes(src0),
                ne00,
                ne01,
                ne02,
                ne03,
                nb01,
                nb02,
                nb03,
                (const float *) src1->data,
                ne11,
                ne12,
                ne13,
                nb11,
                nb12,
                nb13,
                (float *) dst->data,
                nb1,
                nb2,
                nb3);
            atomic_store_explicit(&params->threadpool->current_chunk, done ? 1 : 0, memory_order_relaxed);
        }

        ggml_barrier(params->threadpool);
        if (atomic_load_explicit(&params->threadpool->current_chunk, memory_order_relaxed) != 0) {
            return;
        }
    }

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    // TODO: extract to "extra_op"
#if GGML_USE_LLAMAFILE
    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    const bool src1_cont = ggml_is_contiguous(src1);

    if (src1_cont) {
        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (const char *)src1->data + i12*nb12 + i13*nb13,
                                     nb11/ggml_type_size(src1->type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    goto UseGgmlGemm1;
        return;
    }
UseGgmlGemm1:;
#endif

    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

    #if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
    #else
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
    #endif
    }

    if (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);

#if GGML_USE_LLAMAFILE
    if (src1->type != vec_dot_type) {
        const void* wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (const char *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
                                     row_size/ggml_type_size(vec_dot_type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    goto UseGgmlGemm2;
        return;
    }
UseGgmlGemm2:;
#endif

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    const int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    const int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    int chunk_size = 16;

    // We need to step up the size if it's small
    if (nr0 == 1 || nr1 == 1) {
        chunk_size = 64;
    }

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim.
    // CEIL(nr0/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    // If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
    //   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggml-org/llama.cpp/pull/6915
    //   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.
    if (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    int current_chunk = ith;

    while (current_chunk < nchunk0 * nchunk1) {
        const int64_t ith0 = current_chunk % nchunk0;
        const int64_t ith1 = current_chunk / nchunk0;

        const int64_t ir0_start = dr0 * ith0;
        const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        const int64_t ir1_start = dr1 * ith1;
        const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int64_t num_rows_per_vec_dot = vec_dot_num_rows;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        if ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
            num_rows_per_vec_dot = 1;
        }
        ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

        if (nth >= nchunk0 * nchunk1) {
            break;
        }

        current_chunk = atomic_fetch_add_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
    }
}

// ggml_compute_forward_mul_mat_id

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id)*ids->ne[0]*ids->ne[1] + (i1)]

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

#define GGML_MOE_CPU_BATCH_MICROPROBE_MAX_THREADS 256

static atomic_int ggml_moe_cpu_batch_microprobe_next_seq = 0;
static atomic_int ggml_moe_cpu_batch_microprobe_active_seq = -1;
static atomic_int ggml_moe_cpu_batch_microprobe_active_expert = -1;

static uint64_t ggml_moe_cpu_batch_microprobe_dot_calls[GGML_MOE_CPU_BATCH_MICROPROBE_MAX_THREADS];
static uint64_t ggml_moe_cpu_batch_microprobe_diff_count[GGML_MOE_CPU_BATCH_MICROPROBE_MAX_THREADS];
static uint64_t ggml_moe_cpu_batch_microprobe_src0_bytes[GGML_MOE_CPU_BATCH_MICROPROBE_MAX_THREADS];
static uint64_t ggml_moe_cpu_batch_microprobe_q80_bytes[GGML_MOE_CPU_BATCH_MICROPROBE_MAX_THREADS];
static double   ggml_moe_cpu_batch_microprobe_max_abs[GGML_MOE_CPU_BATCH_MICROPROBE_MAX_THREADS];
static double   ggml_moe_cpu_batch_microprobe_sum_abs[GGML_MOE_CPU_BATCH_MICROPROBE_MAX_THREADS];
static double   ggml_moe_cpu_batch_microprobe_sink[GGML_MOE_CPU_BATCH_MICROPROBE_MAX_THREADS];

static void ggml_compute_forward_mul_mat_id_one_chunk(
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    const int64_t cur_a,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end,
    const char * src0_cur,
    const struct mmid_row_mapping * matrix_rows,
    const size_t row_size,
    const bool src1_cont,
    const void * wdata) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;

    ggml_vec_dot_t    const vec_dot      = type_traits_cpu[type].vec_dot;
    enum ggml_type    const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    float tmp[16];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
                const int64_t _i12 = ir1; // logical row index for this expert

                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
                const int id       = row_mapping.i1; // selected expert index

                const int64_t  i11 = id % ne11;
                const int64_t  i12 = row_mapping.i2; // row index in src1

                const int64_t  i1 = id;  // selected expert index
                const int64_t  i2 = i12; // row

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*row_size
                    : (i11*nb11 + i12*nb12));

                float * dst_col = (float *) ((char *) dst->data + (i1*nb1 + i2*nb2));

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }

                memcpy(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir0_end) - iir0)*sizeof(float));
            }
        }
    }
}

static void ggml_compute_forward_mul_mat_id_microprobe_one_chunk(
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    const int64_t cur_a,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end,
    const char * src0_cur,
    const struct mmid_row_mapping * matrix_rows,
    const size_t row_size,
    const bool src1_cont,
    const void * wdata,
    const int repeats,
    const int ith) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;

    ggml_vec_dot_t const vec_dot = type_traits_cpu[type].vec_dot;
    enum ggml_type const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    uint64_t dot_calls = 0;
    uint64_t diff_count = 0;
    uint64_t src0_bytes = 0;
    uint64_t q80_bytes = 0;
    double max_abs = 0.0;
    double sum_abs = 0.0;
    double sink = 0.0;

    for (int rep = 0; rep < repeats; ++rep) {
        for (int64_t ir1 = ir1_start; ir1 < ir1_end; ++ir1) {
            const int64_t _i12 = ir1;

            struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
            const int id = row_mapping.i1;

            const int64_t i11 = id % ne11;
            const int64_t i12 = row_mapping.i2;

            const int64_t i1 = id;
            const int64_t i2 = i12;

            const char * src1_col = (const char *) wdata +
                (src1_cont || src1->type != vec_dot_type
                ? (i11      + i12*ne11)*row_size
                : (i11*nb11 + i12*nb12));

            const float * dst_col = (const float *) ((const char *) dst->data + (i1*nb1 + i2*nb2));

            for (int64_t ir0 = ir0_start; ir0 < ir0_end; ++ir0) {
                float cpu = 0.0f;
                vec_dot(ne00, &cpu, 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                sink += (double) cpu;
                dot_calls++;
                src0_bytes += (uint64_t) nb01;
                q80_bytes += (uint64_t) row_size;

                if (rep == 0) {
                    const double diff = fabs((double) cpu - (double) dst_col[ir0]);
                    if (diff > max_abs) {
                        max_abs = diff;
                    }
                    sum_abs += diff;
                    diff_count++;
                }
            }
        }
    }

    if (ith >= 0 && ith < GGML_MOE_CPU_BATCH_MICROPROBE_MAX_THREADS) {
        ggml_moe_cpu_batch_microprobe_dot_calls[ith] += dot_calls;
        ggml_moe_cpu_batch_microprobe_diff_count[ith] += diff_count;
        ggml_moe_cpu_batch_microprobe_src0_bytes[ith] += src0_bytes;
        ggml_moe_cpu_batch_microprobe_q80_bytes[ith] += q80_bytes;
        if (max_abs > ggml_moe_cpu_batch_microprobe_max_abs[ith]) {
            ggml_moe_cpu_batch_microprobe_max_abs[ith] = max_abs;
        }
        ggml_moe_cpu_batch_microprobe_sum_abs[ith] += sum_abs;
        ggml_moe_cpu_batch_microprobe_sink[ith] += sink;
    }
}

static void ggml_moe_stream_compare_block_result(
    int seq,
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const int64_t cur_a,
    const char * src0_cur,
    const struct mmid_row_mapping * matrix_rows,
    const size_t row_size,
    const bool src1_cont,
    const void * wdata,
    const int64_t max_k,
    const int32_t max_col) {
    if (src0->type != GGML_TYPE_MXFP4 || src1->type != GGML_TYPE_F32 || max_k < 0 || max_col < 0 || !ggml_moe_stream_compare_block_fp()) {
        return;
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
    const struct mmid_row_mapping row_mapping = matrix_rows[max_k];
    const int id = row_mapping.i1;
    const int64_t i11 = id % ne11;
    const int64_t i12 = row_mapping.i2;
    const char * src1_col_q80 = (const char *) wdata +
        (src1_cont || src1->type != vec_dot_type
        ? (i11 + i12*ne11)*row_size
        : (i11*nb11 + i12*nb12));
    const block_q8_0 * y_wdata = (const block_q8_0 *) src1_col_q80;
    const float * src1_f32 = (const float *) ((const char *) src1->data + i11*nb11 + i12*nb12);
    const block_mxfp4 * x = (const block_mxfp4 *) (src0_cur + (int64_t) max_col*nb01);
    const float gpu = *(const float *) ((const char *) dst->data + (id*nb1 + i12*nb2 + (int64_t) max_col*nb0));

    double cpu_wdata_total = 0.0;
    double post_src1_total = 0.0;

    const int64_t nb = ne00 / QK_MXFP4;
    for (int64_t ib = 0; ib < nb; ++ib) {
        float amax = 0.0f;
        for (int j = 0; j < QK_MXFP4; ++j) {
            amax = MAX(amax, fabsf(src1_f32[ib*QK_MXFP4 + j]));
        }

        const float d_post = amax / ((1 << 7) - 1);
        const float id_q = d_post ? 1.0f/d_post : 0.0f;
        const ggml_fp16_t d_h = GGML_FP32_TO_FP16(d_post);
        const float d_post_q = GGML_FP16_TO_FP32(d_h);

        int8_t q_post[QK_MXFP4];
        for (int j = 0; j < QK_MXFP4; ++j) {
            const int q = (int) roundf(src1_f32[ib*QK_MXFP4 + j] * id_q);
            q_post[j] = (int8_t) q;
        }

        int sumi_wdata = 0;
        int sumi_post = 0;
        for (int j = 0; j < QK_MXFP4/2; ++j) {
            const int v0 = ggml_moe_stream_mxfp4_values[x[ib].qs[j] & 0x0F];
            const int v1 = ggml_moe_stream_mxfp4_values[x[ib].qs[j] >> 4];
            sumi_wdata += y_wdata[ib].qs[j] * v0;
            sumi_wdata += y_wdata[ib].qs[j + QK_MXFP4/2] * v1;
            sumi_post += q_post[j] * v0;
            sumi_post += q_post[j + QK_MXFP4/2] * v1;
        }

        const float scale_base = GGML_E8M0_TO_FP32_HALF(x[ib].e);
        const float d_wdata = GGML_FP16_TO_FP32(y_wdata[ib].d);
        const double cpu_part = (double) scale_base * (double) d_wdata * (double) sumi_wdata;
        const double post_part = (double) scale_base * (double) d_post_q * (double) sumi_post;
        cpu_wdata_total += cpu_part;
        post_src1_total += post_part;

        ggml_moe_stream_compare_block_write(
            seq, src0->name, src0->type, (int) cur_a, max_k, id, i12, max_col, ib,
            cpu_part, post_part, d_wdata, d_post_q, (int) x[ib].e, amax, sumi_wdata, sumi_post,
            cpu_wdata_total, post_src1_total, gpu);
    }
}

static void ggml_moe_stream_compare_cpu_result(
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const int64_t cur_a,
    const int64_t cne1,
    const char * src0_cur,
    const struct mmid_row_mapping * matrix_rows,
    const size_t row_size,
    const bool src1_cont,
    const void * wdata) {
    int seq = 0;
    if (!ggml_moe_stream_compare_cpu_take_record(&seq)) {
        return;
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;
    ggml_vec_dot_t const vec_dot = type_traits_cpu[type].vec_dot;
    enum ggml_type const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    double sum_abs = 0.0;
    double max_abs = -1.0;
    double max_rel = 0.0;
    int64_t count = 0;
    int64_t max_k = -1;
    int32_t max_row = -1;
    int32_t max_col = -1;
    float max_cpu = 0.0f;
    float max_gpu = 0.0f;

    for (int64_t k = 0; k < cne1; ++k) {
        struct mmid_row_mapping row_mapping = matrix_rows[k];
        const int id = row_mapping.i1;
        const int64_t i11 = id % ne11;
        const int64_t i12 = row_mapping.i2;

        const char * src1_col = (const char *) wdata +
            (src1_cont || src1->type != vec_dot_type
            ? (i11 + i12*ne11)*row_size
            : (i11*nb11 + i12*nb12));

        const float * dst_col = (const float *) ((const char *) dst->data + (id*nb1 + i12*nb2));

        for (int64_t ir0 = 0; ir0 < ne01; ++ir0) {
            float cpu = 0.0f;
            vec_dot(ne00, &cpu, 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
            const float gpu = dst_col[ir0];
            const double abs_diff = fabs((double) cpu - (double) gpu);
            const double denom = fmax(fabs((double) cpu), 1.0e-12);
            const double rel_diff = abs_diff / denom;

            sum_abs += abs_diff;
            count += 1;
            if (abs_diff > max_abs) {
                max_abs = abs_diff;
                max_rel = rel_diff;
                max_k = k;
                max_row = id;
                max_col = (int32_t) ir0;
                max_cpu = cpu;
                max_gpu = gpu;
            }
        }
    }

    ggml_moe_stream_compare_cpu_write(
        seq, src0->name, src0->type, (int) cur_a, cne1, count,
        max_abs < 0.0 ? 0.0 : max_abs,
        count > 0 ? sum_abs / (double) count : 0.0,
        max_rel,
        max_k,
        max_row,
        max_col,
        max_cpu,
        max_gpu);

    ggml_moe_stream_compare_block_result(
        seq, dst, src0, src1, cur_a, src0_cur, matrix_rows, row_size, src1_cont, wdata, max_k, max_col);
}

static void * incr_ptr_aligned(void ** p, size_t size, size_t align) {

    void * ptr = *p;
    ptr = (void *) GGML_PAD((uintptr_t) ptr, align);
    *p = (void *) ((char *) ptr + size);
    return ptr;
}

static bool ggml_moe_gpu_provider_enabled(const struct ggml_tensor * ids) {
    const char * arm_file = getenv("GGML_MOE_DECODE_PROVIDER_ARM_FILE");
    if (arm_file && arm_file[0] && access(arm_file, F_OK) != 0) {
        return false;
    }

    if (ids->ne[1] <= 1) {
        return true;
    }
    const char * min_tokens_env = getenv("GGML_MOE_NATIVE_PREFILL_MIN_TOKENS");
    if (min_tokens_env && min_tokens_env[0]) {
        const long min_tokens = strtol(min_tokens_env, NULL, 10);
        if (min_tokens > 1) {
            return ids->ne[1] < min_tokens;
        }
    }
    const char * env = getenv("GGML_MOE_PROMPT_MATMUL_ID_BATCH");
    return !env || !env[0] || env[0] != '0';
}

static bool ggml_kimi_pipeline_v2_active(void) {
    const char * active = getenv("GGML_KIMI_PIPELINE_V2_ACTIVE");
    const char * mode = getenv("GGML_KIMI_PIPELINE_V2_MODE");
    return active && active[0] && active[0] != '0' && mode &&
        (strcmp(mode, "depth1") == 0 || strcmp(mode, "pipeline") == 0);
}

static uint64_t ggml_kimi_pipeline_m1_fnv1a(const void * data, size_t size) {
    const unsigned char * bytes = (const unsigned char *) data;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < size; ++i) {
        hash = (hash ^ bytes[i]) * UINT64_C(1099511628211);
    }
    return hash;
}

static bool ggml_kimi_pipeline_m1_checkpoint_wants(const struct ggml_tensor * tensor) {
    if (!ggml_kimi_pipeline_v2_active() || !tensor || !tensor->data ||
            tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor)) {
        return false;
    }
    const char * path = getenv("GGML_KIMI_PIPELINE_M1_CHECKPOINT_OUT");
    if (!path || !path[0]) {
        return false;
    }
    return strstr(tensor->name, "ffn_moe_situ") != NULL ||
        strstr(tensor->name, "ffn_moe_weighted") != NULL ||
        strstr(tensor->name, "ffn_moe_out") != NULL;
}

static pthread_mutex_t ggml_kimi_pipeline_m1_checkpoint_mu = PTHREAD_MUTEX_INITIALIZER;
static atomic_uint_fast64_t ggml_kimi_pipeline_m1_checkpoint_seq = 0;
static bool ggml_kimi_pipeline_m1_checkpoint_header = false;

static void ggml_kimi_pipeline_m1_checkpoint_record(const struct ggml_tensor * tensor) {
    const char * path = getenv("GGML_KIMI_PIPELINE_M1_CHECKPOINT_OUT");
    if (!path || !path[0] || !tensor || !tensor->data) {
        return;
    }
    const char * run = getenv("GGML_KIMI_PIPELINE_M1_RUN");
    const char * mode = getenv("GGML_KIMI_PIPELINE_V2_MODE");
    const size_t bytes = ggml_nbytes(tensor);
    const uint64_t hash = ggml_kimi_pipeline_m1_fnv1a(tensor->data, bytes);
    const uint64_t seq = atomic_fetch_add_explicit(
        &ggml_kimi_pipeline_m1_checkpoint_seq, 1, memory_order_relaxed);
    pthread_mutex_lock(&ggml_kimi_pipeline_m1_checkpoint_mu);
    FILE * file = fopen(path, "a");
    if (file) {
        if (!ggml_kimi_pipeline_m1_checkpoint_header) {
            fprintf(file, "seq,run,mode,tensor,op,bytes,fnv\n");
            ggml_kimi_pipeline_m1_checkpoint_header = true;
        }
        fprintf(file, "%" PRIu64 ",%s,%s,%s,%s,%zu,%016" PRIx64 "\n",
                seq, run ? run : "", mode ? mode : "", tensor->name,
                ggml_op_name(tensor->op), bytes, hash);
        fclose(file);
    }
    pthread_mutex_unlock(&ggml_kimi_pipeline_m1_checkpoint_mu);
}

static atomic_bool ggml_kimi_a2_request_phase_active = false;
static atomic_uint_fast64_t ggml_kimi_full_coverage_request_seq = 0;
static atomic_uint_fast64_t ggml_kimi_full_coverage_expert_calls = 0;
static atomic_uint_fast64_t ggml_kimi_full_coverage_fallback_tensors = 0;
static atomic_uint_fast64_t ggml_kimi_full_coverage_fallback_rows = 0;
static pthread_mutex_t ggml_kimi_full_coverage_trace_mu = PTHREAD_MUTEX_INITIALIZER;
static bool ggml_kimi_full_coverage_trace_header = false;

static bool ggml_kimi_full_coverage_prefill_enabled(void) {
    const char * env = getenv("GGML_MOE_PREFILL_FULL_COVERAGE");
    return env && env[0] && env[0] != '0';
}

static void ggml_kimi_full_coverage_counters_reset(void) {
    atomic_store_explicit(&ggml_kimi_full_coverage_expert_calls, 0, memory_order_relaxed);
    atomic_store_explicit(&ggml_kimi_full_coverage_fallback_tensors, 0, memory_order_relaxed);
    atomic_store_explicit(&ggml_kimi_full_coverage_fallback_rows, 0, memory_order_relaxed);
}

static void ggml_kimi_full_coverage_report(void) {
    if (!ggml_kimi_full_coverage_prefill_enabled()) {
        return;
    }
    fprintf(stderr,
            "[kimi_full_coverage_prefill] request=%" PRIuFAST64
            " expert_calls=%" PRIuFAST64 " cpu_fallback_expert_tensors=%" PRIuFAST64
            " cpu_fallback_expert_rows=%" PRIuFAST64 "\n",
            atomic_load_explicit(&ggml_kimi_full_coverage_request_seq, memory_order_relaxed),
            atomic_load_explicit(&ggml_kimi_full_coverage_expert_calls, memory_order_relaxed),
            atomic_load_explicit(&ggml_kimi_full_coverage_fallback_tensors, memory_order_relaxed),
            atomic_load_explicit(&ggml_kimi_full_coverage_fallback_rows, memory_order_relaxed));
}

void ggml_cpu_kimi_a2_request_phase(int active);

void ggml_cpu_kimi_a2_request_phase(int active) {
    const bool enabled = active != 0;
    const bool previous = atomic_exchange_explicit(
        &ggml_kimi_a2_request_phase_active, enabled, memory_order_acq_rel);
    if (ggml_kimi_full_coverage_prefill_enabled()) {
        if (enabled && !previous) {
            atomic_fetch_add_explicit(
                &ggml_kimi_full_coverage_request_seq, 1, memory_order_relaxed);
            ggml_kimi_full_coverage_counters_reset();
        } else if (!enabled && previous) {
            ggml_kimi_full_coverage_report();
        }
    }
}

static bool ggml_cpu_kimi_a2_request_phase_is_active(void) {
    return atomic_load_explicit(
        &ggml_kimi_a2_request_phase_active, memory_order_acquire);
}

static void ggml_kimi_full_coverage_cpu_fallback_record(
        const struct ggml_tensor * src0,
        const struct ggml_tensor * ids,
        const int64_t * matrix_row_counts,
        int64_t n_as) {
    if (!ggml_kimi_full_coverage_prefill_enabled() ||
            !ggml_cpu_kimi_a2_request_phase_is_active() ||
            !src0 || !ids || !matrix_row_counts || ids->ne[1] <= 1 || n_as <= 0) {
        return;
    }

    int64_t fallback_rows = 0;
    for (int64_t expert = 0; expert < n_as; ++expert) {
        if (matrix_row_counts[expert] > 0) {
            fallback_rows += matrix_row_counts[expert];
        }
    }
    atomic_fetch_add_explicit(
        &ggml_kimi_full_coverage_expert_calls, 1, memory_order_relaxed);
    if (fallback_rows <= 0) {
        return;
    }

    atomic_fetch_add_explicit(
        &ggml_kimi_full_coverage_fallback_tensors, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(
        &ggml_kimi_full_coverage_fallback_rows, (uint64_t) fallback_rows, memory_order_relaxed);
    fprintf(stderr,
            "[kimi_full_coverage_fallback] tensor=%s tokens=%" PRId64
            " cpu_fallback_expert_rows=%" PRId64 "\n",
            src0->name, ids->ne[1], fallback_rows);

    const char * path = getenv("GGML_MOE_PREFILL_FULL_COVERAGE_FALLBACK_OUT");
    if (!path || !path[0]) {
        return;
    }
    pthread_mutex_lock(&ggml_kimi_full_coverage_trace_mu);
    FILE * file = fopen(path, "a");
    if (file) {
        if (!ggml_kimi_full_coverage_trace_header) {
            fprintf(file, "request,tensor,tokens,cpu_fallback_expert_rows\n");
            ggml_kimi_full_coverage_trace_header = true;
        }
        fprintf(file, "%" PRIuFAST64 ",%s,%" PRId64 ",%" PRId64 "\n",
                atomic_load_explicit(
                    &ggml_kimi_full_coverage_request_seq, memory_order_relaxed),
                src0->name, ids->ne[1], fallback_rows);
        fclose(file);
    }
    pthread_mutex_unlock(&ggml_kimi_full_coverage_trace_mu);
}

static bool ggml_kimi_uses_k3_local_adapter(int64_t n_experts, size_t expert_bytes) {
    return n_experts == 896 &&
        (expert_bytes == 3612672 || expert_bytes == 2150400);
}

#include "kimi.c"
#include "kimi-common-adapter.inc"

static void ggml_compute_forward_mul_mat_id_generic(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * ids = dst->src[2];
    // if(strstr(src0->name,"up")){
    //     printf("%d ",src0->ne[1]*src0->nb[1]);
    // }
    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;
    const bool ds4_sparse_fused_mmvq_membership = ggml_ds4_sparse_fused_mmvq_membership_enabled();

    const enum ggml_type type = src0->type;

    const bool src1_cont = ggml_is_contiguous(src1);

    enum ggml_type    const vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t const from_float      = type_traits_cpu[vec_dot_type].from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    const int n_ids = ids->ne[0]; // n_expert_used
    const int n_as  = ne02;       // n_expert
#if defined(__APPLE__)
    const bool mac_pack_source = ggml_mac_expert_pack_enabled() &&
        (n_as == 384 || n_as == 256) && ids->ne[1] == 1 &&
        strstr(src0->name, ".ffn_") && strstr(src0->name, "_exps.weight");
#endif

    // Select the model adapter before any availability/cache probe can
    // initialize the other model's process-global CUDA state.
    if (ggml_cuda_moe_adapter_select_model_shape) {
        ggml_cuda_moe_adapter_select_model_shape(n_as, (size_t) ne01 * nb01);
    }

    void * wdata_cur = params->wdata;

    if (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_row_size(vec_dot_type, ggml_nelements(src1)), sizeof(int64_t));
    }

    int64_t * matrix_row_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*sizeof(int64_t), sizeof(int64_t));

    struct mmid_row_mapping * matrix_rows = // [n_as][ids->ne[0]*ids->ne[1]]
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping), sizeof(int64_t));

    char (*atomic_current_chunk)[CACHE_LINE_SIZE] = // [n_as]
        incr_ptr_aligned(&wdata_cur, CACHE_LINE_SIZE * n_as, CACHE_LINE_SIZE);

    const void ** fallback_pack_mmap_ptrs =
        incr_ptr_aligned(&wdata_cur, n_as*sizeof(void *), sizeof(void *));

    uint64_t * fallback_touch_us =
        incr_ptr_aligned(&wdata_cur, n_as*sizeof(uint64_t), sizeof(uint64_t));

    int64_t * gpu_tail_row_counts =
        incr_ptr_aligned(&wdata_cur, n_as*sizeof(int64_t), sizeof(int64_t));

    struct mmid_row_mapping * gpu_tail_matrix_rows =
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping), sizeof(int64_t));

    GGML_ASSERT(params->wsize >= (size_t)((char *) wdata_cur - (char *) params->wdata));

    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#else
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
#endif
    }
    if (ith == 0) {
        // initialize matrix_row_counts
        memset(matrix_row_counts, 0, n_as*sizeof(int64_t));
        const int keep_topk_updown = ggml_moe_keep_topk_for_tensor(src0->name);
        const bool prune_updown = keep_topk_updown > 0;

        // group rows by src0 matrix
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

                if (i02 == -1) {
                    memset((char *) dst->data + id * nb1 + iid1 * nb2, 0, (size_t) ne01 * sizeof(float));
                    continue;
                }
                assert(i02 >= 0 && i02 < n_as);

                if (prune_updown && id >= keep_topk_updown) {
                    memset((char *) dst->data + id * nb1 + iid1 * nb2, 0, (size_t) ne01 * sizeof(float));
                    continue;
                }

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (struct mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }
        ggml_ds4_grouped_retained_route_profile_record(
                src0->name,
                src0->type,
                src0->data,
                nb02,
                ids->ne[1] > 1,
                matrix_row_counts,
                n_as,
                (size_t)ne01 * nb01);
        ggml_ds4_grouped_retained_handoff_record_down(
                src0->name,
                src1->data,
                ne10,
                ne01,
                matrix_row_counts,
                n_as);
    }

    // reset current_chunk
    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    ggml_barrier(params->threadpool);

    const bool kimi_pipeline_v2 = ggml_kimi_pipeline_v2_active();

    if (!kimi_pipeline_v2 &&
            ggml_moe_gpu_provider_enabled(ids) &&
            ggml_cuda_moe_stream_decode_batch &&
            ggml_cuda_moe_stream_available &&
            ggml_cuda_moe_stream_available() &&
            src1->type == GGML_TYPE_F32 &&
            ne13 == 1 &&
            dst->type == GGML_TYPE_F32) {
        if (ith == 0) {
            if (ggml_cuda_moe_stream_route_group_preload_updown_from_gate &&
                    src0->name && strstr(src0->name, ".ffn_gate_exps.") != NULL) {
                const int updown_prefetch_jobs = ggml_cuda_moe_stream_route_group_preload_updown_from_gate(
                        src0->name,
                        n_as,
                        matrix_row_counts);
                (void) updown_prefetch_jobs;
            }
         
            const bool done = ggml_cuda_moe_stream_decode_batch(
                src0->type,
                src0->name,
                src0->data,
                n_as,
                ne01, ne00, nb01, nb02,
                (const float *) src1->data,
                ne11,
                nb11, nb12,
                src1->type == vec_dot_type ? src1->data : params->wdata,
                ggml_row_size(vec_dot_type, ne10),
                (float *) dst->data,
                nb1, nb2,
                matrix_row_counts,
                (const ggml_moe_stream_row_mapping *) matrix_rows,
                ids->ne[0]*ids->ne[1]);
            if (done) {
                if (ggml_moe_stream_compare_cpu_enabled()) {
                    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                        const int64_t cne1 = matrix_row_counts[cur_a];
                        if (cne1 == 0) {
                            continue;
                        }
                        const char * src0_cur = (const char *) src0->data + cur_a * nb02;
                        ggml_moe_stream_compare_cpu_result(
                            dst, src0, src1, cur_a, cne1,
                            src0_cur,
                            matrix_rows + cur_a * ids->ne[0] * ids->ne[1],
                            ggml_row_size(type_traits_cpu[src0->type].vec_dot_type, ne10),
                            src1_cont,
                            src1->type == type_traits_cpu[src0->type].vec_dot_type ? src1->data : params->wdata);
                    }
                }
                memset(matrix_row_counts, 0, n_as*sizeof(int64_t));
            }
        }

        ggml_barrier(params->threadpool);
        int64_t gpu_decode_batch_remaining_rows = 0;
        for (int cur_a = 0; cur_a < n_as; ++cur_a) {
            gpu_decode_batch_remaining_rows += matrix_row_counts[cur_a];
        }
        if (gpu_decode_batch_remaining_rows == 0) {
            return;
        }
    }
    // Decode batch is the only GPU MoE path. Any residual rows fall back to CPU.
    const char * gpu_stream_batch_reason = kimi_pipeline_v2 ?
        "kimi_pipeline_v2" : "decode_batch_only_cpu_fallback";
    const bool use_gpu_stream_batch =
        kimi_pipeline_v2 &&
        ggml_moe_gpu_provider_enabled(ids) &&
        ggml_cuda_moe_stream_batch &&
        ggml_cuda_moe_stream_available &&
        ggml_cuda_moe_stream_available() &&
        n_as == 896 && n_ids == 16 && ids->ne[1] == 1 &&
        src1->type == GGML_TYPE_F32 && ne13 == 1 &&
        dst->type == GGML_TYPE_F32;
    bool gpu_stream_batch_done = false;
    int64_t gpu_stream_single_attempts = 0;
    int64_t gpu_stream_single_accepts = 0;
    uint64_t t0 = ggml_time_us();

    if (kimi_pipeline_v2 && !use_gpu_stream_batch) {
        fprintf(stderr,
                "[kimi_pipeline_m1] fatal guard failure: tensor=%s n_as=%d topk=%d tokens=%" PRId64
                " src0_type=%d src1_type=%d ne13=%" PRId64 " dst_type=%d\n",
                src0->name, n_as, n_ids, ids->ne[1], (int) src0->type,
                (int) src1->type, ne13, (int) dst->type);
        abort();
    }
    // Q8_0 uses the generic one-expert MMVQ transport. The newer decode
    // batch provider currently specializes MXFP4 and can decline this type.
    const char * q8_stream_env = getenv("GGML_MOE_STREAM_Q8_0");
    const bool q8_stream = src0->type == GGML_TYPE_Q8_0 && q8_stream_env &&
        q8_stream_env[0] && q8_stream_env[0] != '0' && ggml_moe_gpu_provider_enabled(ids) &&
        ggml_cuda_moe_stream_available && ggml_cuda_moe_stream_available();
    const char * float_stream_env = getenv("GGML_MOE_STREAM_FLOAT");
    const bool float_stream = (src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F16) &&
        float_stream_env && float_stream_env[0] && float_stream_env[0] != '0' && ggml_moe_gpu_provider_enabled(ids) &&
        ggml_cuda_moe_stream_available && ggml_cuda_moe_stream_available();
    if (!use_gpu_stream_batch && !q8_stream && !float_stream) {
        goto cpu_fallback;
    }
    if (use_gpu_stream_batch) {
        if (ith == 0) {
            const int gpu_keep_topk =
                ggml_moe_gpu_keep_topk_applies(src0->name) ? ggml_moe_gpu_keep_topk_updown() : 0;
            const bool split_gpu_cpu_tail =
                gpu_keep_topk > 0 && gpu_keep_topk < ids->ne[0];
            if (split_gpu_cpu_tail) {
                memset(gpu_tail_row_counts, 0, n_as*sizeof(int64_t));
                for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                    const int64_t original_count = matrix_row_counts[cur_a];
                    int64_t gpu_count = 0;
                    int64_t tail_count = 0;
                    for (int64_t k = 0; k < original_count; ++k) {
                        const struct mmid_row_mapping row =
                            matrix_rows[cur_a * ids->ne[0] * ids->ne[1] + k];
                        if (row.i1 < gpu_keep_topk) {
                            matrix_rows[cur_a * ids->ne[0] * ids->ne[1] + gpu_count++] = row;
                        } else {
                            gpu_tail_matrix_rows[cur_a * ids->ne[0] * ids->ne[1] + tail_count++] = row;
                        }
                    }
                    matrix_row_counts[cur_a] = gpu_count;
                    gpu_tail_row_counts[cur_a] = tail_count;
                }
            }
            const void * src1_q8_0_batch = (src1->type == vec_dot_type) ? src1->data : params->wdata;
            const size_t src1_q8_0_row_size_batch = ggml_row_size(vec_dot_type, ne10);
            // printf("c:%s ",src0->name);
           
            const bool done = ggml_cuda_moe_stream_batch(
                src0->type,
                src0->name,
                src0->data,
                n_as,
                ne01, ne00, nb01, nb02,
                (const float *) src1->data,
                ne11,
                nb11, nb12,
                src1_q8_0_batch,
                src1_q8_0_row_size_batch,
                ne11,
                (float *) dst->data,
                nb1, nb2,
                matrix_row_counts,
                (const ggml_moe_stream_row_mapping *) matrix_rows,
                ids->ne[0]*ids->ne[1]);
            gpu_stream_batch_done = done;

            if (kimi_pipeline_v2 && !done) {
                fprintf(stderr,
                        "[kimi_pipeline_m1] fatal provider decline: tensor=%s type=%d n_as=%d topk=%d\n",
                        src0->name, (int) src0->type, n_as, n_ids);
                abort();
            }

            if (done) {
                if (ggml_moe_stream_compare_cpu_enabled()) {
                    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                        const int64_t cne1 = matrix_row_counts[cur_a];
                        if (cne1 == 0) {
                            continue;
                        }
                        const char * src0_cur = (const char *) src0->data + cur_a * nb02;
                        ggml_moe_stream_compare_cpu_result(
                            dst, src0, src1, cur_a, cne1,
                            src0_cur,
                            matrix_rows + cur_a * ids->ne[0] * ids->ne[1],
                            ggml_row_size(type_traits_cpu[src0->type].vec_dot_type, ne10),
                            src1_cont,
                            src1->type == type_traits_cpu[src0->type].vec_dot_type ? src1->data : params->wdata);
                    }
                }
                if (split_gpu_cpu_tail) {
                    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                        const int64_t tail_count = gpu_tail_row_counts[cur_a];
                        for (int64_t k = 0; k < tail_count; ++k) {
                            matrix_rows[cur_a * ids->ne[0] * ids->ne[1] + k] =
                                gpu_tail_matrix_rows[cur_a * ids->ne[0] * ids->ne[1] + k];
                        }
                        matrix_row_counts[cur_a] = tail_count;
                    }
                } else {
                    memset(matrix_row_counts, 0, n_as*sizeof(int64_t));
                }
            } else if (split_gpu_cpu_tail) {
                for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                    const int64_t gpu_count = matrix_row_counts[cur_a];
                    const int64_t tail_count = gpu_tail_row_counts[cur_a];
                    for (int64_t k = 0; k < tail_count; ++k) {
                        matrix_rows[cur_a * ids->ne[0] * ids->ne[1] + gpu_count + k] =
                            gpu_tail_matrix_rows[cur_a * ids->ne[0] * ids->ne[1] + k];
                    }
                    matrix_row_counts[cur_a] = gpu_count + tail_count;
                }
            }
        }

        ggml_barrier(params->threadpool);
    }

    const bool use_gpu_stream =
        !use_gpu_stream_batch &&
        getenv("GGML_MOE_STREAM_BATCH_ONLY") == NULL &&
        ggml_cuda_moe_stream_one &&
        ggml_cuda_moe_stream_available &&
        ggml_cuda_moe_stream_available() &&
        ggml_cuda_moe_stream_supports_one_type(src0->type) &&
        src1->type == GGML_TYPE_F32 &&
        ne13 == 1 &&
        dst->type == GGML_TYPE_F32;

    if (use_gpu_stream) {
        if (ith == 0) {
            const void * wdata_stream = (src1->type == vec_dot_type) ? src1->data : params->wdata;
            const size_t row_size_stream = ggml_row_size(vec_dot_type, ne10);
            const char * route_group_native_parity_env = getenv("GGML_MOE_ROUTE_GROUP_NATIVE_PARITY");
            const bool route_group_native_parity =
                route_group_native_parity_env && route_group_native_parity_env[0] && route_group_native_parity_env[0] != '0';
            if (ggml_moe_gate_batch_prefetch_enabled() &&
                    !route_group_native_parity &&
                    ggml_cuda_moe_stream_batch_preload_active_from_pack &&
                    src0->name && strstr(src0->name, ".ffn_gate_exps.") != NULL) {
                static bool gate_batch_prefetch_logged = false;
                if (!gate_batch_prefetch_logged) {
                    fprintf(stderr, "[moe_stream_cpu] gate batch prefetch requested: tensor=%s n_as=%" PRId64 " expert_bytes=%zu\n",
                            src0->name, (int64_t) n_as, (size_t) ne01 * nb01);
                    gate_batch_prefetch_logged = true;
                }
                const int gate_prefetch_jobs = ggml_cuda_moe_stream_batch_preload_active_from_pack(
                        src0->type,
                        src0->name,
                        n_as,
                        (size_t) ne01 * nb01,
                        matrix_row_counts);
                (void) gate_prefetch_jobs;
            }
            if (ggml_cuda_moe_stream_route_group_preload_updown_from_gate &&
                    src0->name && strstr(src0->name, ".ffn_gate_exps.") != NULL) {
                const int updown_prefetch_jobs = ggml_cuda_moe_stream_route_group_preload_updown_from_gate(
                        src0->name,
                        n_as,
                        matrix_row_counts);
                (void) updown_prefetch_jobs;
            }

          
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                const int64_t cne1 = matrix_row_counts[cur_a];

                if (cne1 == 0) {
                    continue;
                }
                //这里只执行6次

                const char * src0_cur = (const char *) src0->data + cur_a * nb02;
                gpu_stream_single_attempts++;
                bool done = true;
                const int64_t chunk_limit = (q8_stream || float_stream) ? 8 : cne1;
                for (int64_t first = 0; first < cne1; first += chunk_limit) {
                    const int64_t chunk = MIN(chunk_limit, cne1 - first);
                    const bool chunk_done = ggml_cuda_moe_stream_one(
                        src0->type,
                        src0->name,
                        cur_a,
                        src0_cur,
                        ne01, ne00, nb01,
                        (const float *) src1->data,
                        nb11, nb12,
                        ne11,
                        chunk,
                        NULL, 0,
                        (float *) dst->data,
                        nb1, nb2,
                        (const ggml_moe_stream_row_mapping *) (matrix_rows + cur_a * ids->ne[0] * ids->ne[1] + first));
                    if (!chunk_done) { done = false; break; }
                }


                if (done) {
                    gpu_stream_single_accepts++;
                    if (ggml_moe_stream_compare_cpu_enabled()) {
                        ggml_moe_stream_compare_cpu_result(
                            dst, src0, src1, cur_a, cne1,
                            src0_cur,
                            matrix_rows + cur_a * ids->ne[0] * ids->ne[1],
                            ggml_row_size(type_traits_cpu[src0->type].vec_dot_type, ne10),
                            src1_cont,
                            src1->type == type_traits_cpu[src0->type].vec_dot_type ? src1->data : params->wdata);
                    }
                    matrix_row_counts[cur_a] = 0;
                }
            }

            if (ggml_cuda_moe_stream_sync) {
                ggml_cuda_moe_stream_sync();
            }
        }

        ggml_barrier(params->threadpool);

        if (ggml_moe_stream_one_fast_return_enabled()) {
            // If every routed row was accepted by ggml_moe_stream_one, skip the CPU fallback/probe tail.
            int64_t gpu_stream_remaining_rows = 0;
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                gpu_stream_remaining_rows += matrix_row_counts[cur_a];
            }
            if (gpu_stream_remaining_rows == 0) {
                return;
            }
        }
    }
    uint64_t t1 = ggml_time_us();
    if (ggml_moe_stream_one_gpu_only_filter_matches(src0->name)) {
        //gate up down都不会走这部分
    
        int64_t gpu_only_rows = 0;
        for (int cur_a = 0; cur_a < n_as; ++cur_a) {
            gpu_only_rows += matrix_row_counts[cur_a];
        }
        if (gpu_only_rows > 0) {
            fprintf(stderr,
                    "[moe_stream] GPU-only filter matched but CPU fallback remains: tensor=%s rows=%" PRId64 " batch_reason=%s single_attempts=%" PRId64 " single_accepts=%" PRId64 "\n",
                    src0->name ? src0->name : "", gpu_only_rows,
                    gpu_stream_batch_reason,
                    gpu_stream_single_attempts, gpu_stream_single_accepts);
            abort();
        }
    }

cpu_fallback:
    if (ggml_moe_cpu_willneed_enabled()) {
        //都不会走
        if (ith == 0) {
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                if (matrix_row_counts[cur_a] == 0) {
                    continue;
                }
                ggml_moe_cpu_willneed_pages((const char *) src0->data + cur_a * nb02, (size_t) ne01 * nb01);
            }
        }
        ggml_barrier(params->threadpool);
    }

    if (ith == 0) {
        //会走
#if defined(__APPLE__)
        if (mac_pack_source) {
            memset(fallback_pack_mmap_ptrs, 0, (size_t) n_as * sizeof(void *));
        } else
#endif
        {
            const char * ds4_handoff_env = getenv("GGML_MOE_STREAM_DS4_TRIPLET_CPU_DOWN_HANDOFF");
            const bool ds4_handoff_enabled = ds4_handoff_env && ds4_handoff_env[0] && ds4_handoff_env[0] != '0';
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                fallback_pack_mmap_ptrs[cur_a] = NULL;
                if (ds4_handoff_enabled && matrix_row_counts[cur_a] > 0 &&
                        ggml_cuda_moe_stream_ds4_triplet_host_role_ptr) {
                    fallback_pack_mmap_ptrs[cur_a] =
                    ggml_cuda_moe_stream_ds4_triplet_host_role_ptr(
                        src0->name, cur_a, (size_t) ne01 * nb01);
                }
            }
        }
        memset(fallback_touch_us, 0, n_as*sizeof(uint64_t));
        if (ggml_moe_cpu_fallback_touch_profile_enabled()) {
            //不会走
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                if (matrix_row_counts[cur_a] == 0) {
                    continue;
                }
                const char * src0_cur = fallback_pack_mmap_ptrs[cur_a] ?
                    (const char *) fallback_pack_mmap_ptrs[cur_a] :
                    (const char *) src0->data + cur_a * nb02;
                fallback_touch_us[cur_a] =
                    ggml_moe_cpu_touch_pages_us(src0_cur, (size_t) ne01 * nb01);
            }
        }
    }
    ggml_barrier(params->threadpool);

    const bool use_cpu_batch_microprobe =
        ggml_moe_cpu_batch_microprobe_enabled() &&
        src0->type == GGML_TYPE_MXFP4 &&
        src1->type != vec_dot_type &&
        vec_dot_type == GGML_TYPE_Q8_0 &&
        ne13 == 1 &&
        dst->type == GGML_TYPE_F32 &&
        nth <= GGML_MOE_CPU_BATCH_MICROPROBE_MAX_THREADS &&
        ggml_moe_cpu_batch_microprobe_name_allows(src0->name);

    int cpu_batch_microprobe_cur_a = -1;
    int cpu_batch_microprobe_seq = -1;
    if (use_cpu_batch_microprobe) {
        if (ith == 0) {
            int selected = -1;
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                if (matrix_row_counts[cur_a] > 0) {
                    selected = cur_a;
                    break;
                }
            }

            const int seq = atomic_fetch_add_explicit(&ggml_moe_cpu_batch_microprobe_next_seq, 1, memory_order_relaxed);
            const int max_calls = ggml_moe_cpu_batch_microprobe_env_int("GGML_MOE_CPU_BATCH_MICROPROBE_MAX_CALLS", 1);
            if (selected >= 0 && (max_calls <= 0 || seq < max_calls)) {
                atomic_store_explicit(&ggml_moe_cpu_batch_microprobe_active_seq, seq, memory_order_relaxed);
                atomic_store_explicit(&ggml_moe_cpu_batch_microprobe_active_expert, selected, memory_order_relaxed);
            } else {
                atomic_store_explicit(&ggml_moe_cpu_batch_microprobe_active_seq, -1, memory_order_relaxed);
                atomic_store_explicit(&ggml_moe_cpu_batch_microprobe_active_expert, -1, memory_order_relaxed);
            }
        }
        ggml_barrier(params->threadpool);
        cpu_batch_microprobe_seq = atomic_load_explicit(&ggml_moe_cpu_batch_microprobe_active_seq, memory_order_relaxed);
        cpu_batch_microprobe_cur_a = atomic_load_explicit(&ggml_moe_cpu_batch_microprobe_active_expert, memory_order_relaxed);
    }

    if (ggml_moe_stream_q80_skip_enabled() &&
            ggml_cuda_moe_stream_q80_skip &&
            src0->type == GGML_TYPE_MXFP4 &&
            src1->type != vec_dot_type &&
            vec_dot_type == GGML_TYPE_Q8_0 &&
            ne13 == 1 &&
            dst->type == GGML_TYPE_F32) {
        if (ith == 0) {
            const void * q80_base = params->wdata;
            const size_t q80_row_size = ggml_row_size(vec_dot_type, ne10);
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                const int64_t cne1 = matrix_row_counts[cur_a];
                if (cne1 == 0) {
                    continue;
                }
                const char * src0_cur = fallback_pack_mmap_ptrs[cur_a] ?
                    (const char *) fallback_pack_mmap_ptrs[cur_a] :
                    (const char *) src0->data + cur_a * nb02;
                const bool done = ggml_cuda_moe_stream_q80_skip(
                        src0->type,
                        src0->name,
                        cur_a,
                        src0_cur,
                        ne01, ne00, nb01,
                        q80_base,
                        q80_row_size,
                        ne11,
                        cne1,
                        (float *) dst->data,
                        nb1, nb2,
                        (const ggml_moe_stream_row_mapping *) (matrix_rows + cur_a * ids->ne[0] * ids->ne[1]));
                if (done) {
                    matrix_row_counts[cur_a] = 0;
                }
            }
        }
        ggml_barrier(params->threadpool);
    }

    const bool moe_fallback_reason_profile = ggml_moe_fallback_reason_profile_enabled();
    // const uint64_t kimi_cpu_moe_fallback_start = (kimi_cpu_moe_profile && ith == 0) ? ggml_time_us() : 0;
    const uint64_t moe_fallback_reason_start = (moe_fallback_reason_profile && ith == 0) ? ggml_time_us() : 0;
    const uint64_t ds4_sparse_fused_mmvq_fallback_start =
        (ds4_sparse_fused_mmvq_membership && ith == 0) ? ggml_time_us() : 0;
#if defined(__APPLE__)
    void * mac_read_batch = NULL;
    if (mac_pack_source && ith == 0) {
        mac_read_batch = ggml_mac_expert_pack_begin_batch(src0->name, matrix_row_counts, n_as, (size_t) ne01 * nb01);
    }
    uint64_t mac_compute_start = 0, mac_compute_us = 0, mac_barrier_start = 0, mac_barrier_us = 0;
#endif
    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = matrix_row_counts[cur_a];
        if (cne1 == 0) {
            continue;
        }
#if defined(__APPLE__)
        if (mac_pack_source) {
            if (ith == 0) {
                fallback_pack_mmap_ptrs[cur_a] =
                    ggml_mac_expert_pack_acquire_batch(mac_read_batch, src0->name, cur_a, (size_t) ne01 * nb01);
                mac_barrier_start = ggml_time_us();
            }
            ggml_barrier(params->threadpool);
            if (ith == 0) {
                mac_barrier_us = ggml_time_us() - mac_barrier_start;
                mac_compute_start = ggml_time_us();
            }
        }
#endif
        // printf("k=");
        const char * src0_cur = fallback_pack_mmap_ptrs[cur_a] ?
            (const char *) fallback_pack_mmap_ptrs[cur_a] :
            (const char *) src0->data + cur_a * nb02;
        const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        const int64_t nr0 = ne01;
        const int64_t nr1 = cne1;

        int chunk_size = 16;
        if (nr0 == 1 || nr1 == 1) {
            chunk_size = 64;
        }

        // disable for NUMA
        const bool disable_chunking = ggml_is_numa();

        int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

        if (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        int current_chunk = ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        while (current_chunk < nchunk0 * nchunk1) {
            const int64_t ith0 = current_chunk % nchunk0;
            const int64_t ith1 = current_chunk / nchunk0;

            const int64_t ir0_start = dr0 * ith0;
            const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

            const int64_t ir1_start = dr1 * ith1;
            const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

            const bool trace_cpu_chunk = ggml_moe_cpu_chunk_trace_enabled();
            const double trace_t0_ms = trace_cpu_chunk ? ggml_moe_cpu_trace_now_ms() : 0.0;

            ggml_compute_forward_mul_mat_id_one_chunk(
                dst, src0, src1, ids, cur_a,
                ir0_start, ir0_end, ir1_start, ir1_end,
                src0_cur, matrix_rows, row_size, src1_cont, wdata
            );

            if (trace_cpu_chunk) {
               //不会执行
                const double trace_t1_ms = ggml_moe_cpu_trace_now_ms();
                ggml_moe_cpu_chunk_trace_write(
                    ggml_moe_tensor_role(src0->name),
                    src0->name,
                    src0->type,
                    cur_a,
                    ith,
                    nth,
                    cne1,
                    ir0_start,
                    ir0_end,
                    ir1_start,
                    ir1_end,
                    (size_t) ne01 * nb01,
                    trace_t1_ms - trace_t0_ms);
            }

            if (nth >= nchunk0 * nchunk1) {
                break;
            }

            current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }

        if (cpu_batch_microprobe_cur_a == cur_a) {
            const int repeats_env = ggml_moe_cpu_batch_microprobe_env_int("GGML_MOE_CPU_BATCH_MICROPROBE_REPEATS", 1);
            const int repeats = repeats_env > 0 ? repeats_env : 1;
            const int64_t max_cols_env = (int64_t) ggml_moe_cpu_batch_microprobe_env_int("GGML_MOE_CPU_BATCH_MICROPROBE_MAX_COLS", 0);
            const int64_t max_rows_env = (int64_t) ggml_moe_cpu_batch_microprobe_env_int("GGML_MOE_CPU_BATCH_MICROPROBE_MAX_ROWS", 1);
            const int64_t probe_cols = max_cols_env > 0 ? MIN(nr0, max_cols_env) : nr0;
            const int64_t probe_rows = max_rows_env > 0 ? MIN(nr1, max_rows_env) : nr1;

            ggml_barrier(params->threadpool);
            if (ith == 0) {
                *current_chunk_ctr = nth;
            }
            if (ith >= 0 && ith < GGML_MOE_CPU_BATCH_MICROPROBE_MAX_THREADS) {
                ggml_moe_cpu_batch_microprobe_dot_calls[ith] = 0;
                ggml_moe_cpu_batch_microprobe_diff_count[ith] = 0;
                ggml_moe_cpu_batch_microprobe_src0_bytes[ith] = 0;
                ggml_moe_cpu_batch_microprobe_q80_bytes[ith] = 0;
                ggml_moe_cpu_batch_microprobe_max_abs[ith] = 0.0;
                ggml_moe_cpu_batch_microprobe_sum_abs[ith] = 0.0;
                ggml_moe_cpu_batch_microprobe_sink[ith] = 0.0;
            }
            ggml_barrier(params->threadpool);

            uint64_t microprobe_start_us = 0;
            if (ith == 0) {
                microprobe_start_us = ggml_time_us();
            }

            current_chunk = ith;
            while (current_chunk < nchunk0 * nchunk1) {
                const int64_t ith0 = current_chunk % nchunk0;
                const int64_t ith1 = current_chunk / nchunk0;

                const int64_t ir0_start_probe = dr0 * ith0;
                const int64_t ir0_end_probe = MIN(ir0_start_probe + dr0, probe_cols);

                const int64_t ir1_start_probe = dr1 * ith1;
                const int64_t ir1_end_probe = MIN(ir1_start_probe + dr1, probe_rows);

                if (ir0_start_probe < ir0_end_probe && ir1_start_probe < ir1_end_probe) {
                    ggml_compute_forward_mul_mat_id_microprobe_one_chunk(
                        dst, src0, src1, ids, cur_a,
                        ir0_start_probe, ir0_end_probe, ir1_start_probe, ir1_end_probe,
                        src0_cur, matrix_rows, row_size, src1_cont, wdata, repeats, ith
                    );
                }

                if (nth >= nchunk0 * nchunk1) {
                    break;
                }

                current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
            }

            ggml_barrier(params->threadpool);

            if (ith == 0) {
                const uint64_t wall_us = ggml_time_us() - microprobe_start_us;
                uint64_t dot_calls = 0;
                uint64_t diff_count = 0;
                uint64_t src0_bytes = 0;
                uint64_t q80_bytes = 0;
                double max_abs = 0.0;
                double sum_abs = 0.0;
                double sink = 0.0;
                for (int it = 0; it < nth && it < GGML_MOE_CPU_BATCH_MICROPROBE_MAX_THREADS; ++it) {
                    dot_calls += ggml_moe_cpu_batch_microprobe_dot_calls[it];
                    diff_count += ggml_moe_cpu_batch_microprobe_diff_count[it];
                    src0_bytes += ggml_moe_cpu_batch_microprobe_src0_bytes[it];
                    q80_bytes += ggml_moe_cpu_batch_microprobe_q80_bytes[it];
                    if (ggml_moe_cpu_batch_microprobe_max_abs[it] > max_abs) {
                        max_abs = ggml_moe_cpu_batch_microprobe_max_abs[it];
                    }
                    sum_abs += ggml_moe_cpu_batch_microprobe_sum_abs[it];
                    sink += ggml_moe_cpu_batch_microprobe_sink[it];
                }
                ggml_moe_cpu_batch_microprobe_write(
                    cpu_batch_microprobe_seq,
                    ggml_moe_tensor_role(src0->name),
                    src0->name,
                    src0->type,
                    cur_a,
                    nth,
                    repeats,
                    cne1,
                    probe_cols,
                    probe_rows,
                    dot_calls,
                    diff_count,
                    src0_bytes,
                    q80_bytes,
                    wall_us,
                    max_abs,
                    sum_abs,
                    sink);
            }
            ggml_barrier(params->threadpool);
        }
#if defined(__APPLE__)
        if (mac_pack_source) {
            if (ith == 0) {
                mac_barrier_start = ggml_time_us();
                mac_compute_us = mac_barrier_start - mac_compute_start;
            }
            ggml_barrier(params->threadpool);
            if (ith == 0) {
                mac_barrier_us += ggml_time_us() - mac_barrier_start;
                ggml_mac_expert_pack_release(fallback_pack_mmap_ptrs[cur_a]);
                fallback_pack_mmap_ptrs[cur_a] = NULL;
                mac_barrier_start = ggml_time_us();
            }
            ggml_barrier(params->threadpool);
            if (ith == 0) {
                mac_barrier_us += ggml_time_us() - mac_barrier_start;
                ggml_mac_expert_pack_record_compute(mac_compute_us, mac_barrier_us);
            }
        }
#endif
    }
#if defined(__APPLE__)
    if (mac_pack_source && ith == 0) { ggml_mac_expert_pack_end_batch(mac_read_batch); }
#endif

    if (moe_fallback_reason_profile && ith == 0) {
        const uint64_t moe_fallback_reason_us = ggml_time_us() - moe_fallback_reason_start;
        int64_t fallback_rows = 0;
        for (int cur_a = 0; cur_a < n_as; ++cur_a) {
            fallback_rows += matrix_row_counts[cur_a];
        }
        const char * batch_reason_name = gpu_stream_batch_reason;
        const char * single_reason_name = "not_attempted";
        const char * final_reason_name = "cpu_fallback";
        if (use_gpu_stream_batch) {
            single_reason_name = "not_attempted_batch_path_selected";
            final_reason_name = gpu_stream_batch_done ? "post_batch_residual" : "batch_declined_internal_no_single_retry";
        } else if (getenv("GGML_MOE_STREAM_BATCH_ONLY") != NULL) {
            single_reason_name = "stream_batch_only";
            final_reason_name = "batch_only_cpu_fallback";
        } else if (!ggml_cuda_moe_stream_one) {
            single_reason_name = "one_fn_missing";
            final_reason_name = "one_fn_missing";
        } else if (!ggml_cuda_moe_stream_available) {
            single_reason_name = "available_fn_missing";
            final_reason_name = "available_fn_missing";
        } else if (!ggml_cuda_moe_stream_available()) {
            single_reason_name = "stream_available_false";
            final_reason_name = "stream_available_false";
        } else if (!ggml_cuda_moe_stream_supports_one_type(src0->type)) {
            single_reason_name = "one_unsupported_type";
            final_reason_name = "one_unsupported_type";
        } else if (src1->type != GGML_TYPE_F32) {
            single_reason_name = "src1_not_f32";
            final_reason_name = "src1_not_f32";
        } else if (ne13 != 1) {
            single_reason_name = "ne13_not1";
            final_reason_name = "ne13_not1";
        } else if (dst->type != GGML_TYPE_F32) {
            single_reason_name = "dst_not_f32";
            final_reason_name = "dst_not_f32";
        } else if (!ggml_moe_stream_one_name_filter_would_allow(src0->name)) {
            single_reason_name = "one_name_filter";
            final_reason_name = "one_name_filter";
        } else {
            single_reason_name = "one_declined_internal";
            final_reason_name = "one_declined_internal";
        }
        if (fallback_rows > 0) {
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                const int64_t cne1 = matrix_row_counts[cur_a];
                if (cne1 == 0) {
                    continue;
                }
                const uint64_t expert_fallback_us =
                    (uint64_t) (((double) moe_fallback_reason_us * (double) cne1) / (double) fallback_rows);
                ggml_moe_fallback_reason_profile_record(
                        ggml_moe_tensor_role(src0->name),
                        src0->name,
                        src0->type,
                        ids->ne[1] > 1,
                        cur_a,
                        cne1,
                        (size_t) nb02,
                        expert_fallback_us,
                        batch_reason_name,
                        single_reason_name,
                        final_reason_name,
                        use_gpu_stream_batch,
                        gpu_stream_batch_done,
                        gpu_stream_single_attempts > 0,
                        false);
                ggml_moe_fallback_source_probe_record(
                        ggml_moe_tensor_role(src0->name),
                        src0->name,
                        ids->ne[1] > 1,
                        cur_a,
                        cne1,
                        expert_fallback_us,
                        batch_reason_name,
                        single_reason_name,
                        final_reason_name,
                        src0,
                        src1,
                        ids,
                        dst);
            }
        }
    }
    if (ds4_sparse_fused_mmvq_membership && ith == 0) {
        const uint64_t ds4_sparse_fused_mmvq_fallback_us =
            ggml_time_us() - ds4_sparse_fused_mmvq_fallback_start;
        int64_t fallback_rows = 0;
        for (int cur_a = 0; cur_a < n_as; ++cur_a) {
            fallback_rows += matrix_row_counts[cur_a];
        }
        if (fallback_rows > 0) {
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                const int64_t cne1 = matrix_row_counts[cur_a];
                if (cne1 == 0) {
                    continue;
                }
                const uint64_t expert_fallback_us =
                    (uint64_t) (((double) ds4_sparse_fused_mmvq_fallback_us * (double) cne1) / (double) fallback_rows);
                ggml_ds4_sparse_fused_mmvq_membership_record(
                        src0->name,
                        src0->type,
                        ids->ne[1] > 1,
                        cur_a,
                        cne1,
                        (size_t) nb02,
                        expert_fallback_us,
                        fallback_touch_us[cur_a],
                        fallback_pack_mmap_ptrs[cur_a] != NULL);
            }
        }
    }

    if (ggml_moe_stream_q80_probe_enabled() &&
            ggml_cuda_moe_stream_q80_probe &&
            src0->type == GGML_TYPE_MXFP4 &&
            src1->type != vec_dot_type &&
            vec_dot_type == GGML_TYPE_Q8_0 &&
            ne13 == 1 &&
            dst->type == GGML_TYPE_F32) {
        ggml_barrier(params->threadpool);
        if (ith == 0) {
            const void * q80_base = params->wdata;
            const size_t q80_row_size = ggml_row_size(vec_dot_type, ne10);
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                const int64_t cne1 = matrix_row_counts[cur_a];
                if (cne1 == 0) {
                    continue;
                }
                const char * src0_cur = fallback_pack_mmap_ptrs[cur_a] ?
                    (const char *) fallback_pack_mmap_ptrs[cur_a] :
                    (const char *) src0->data + cur_a * nb02;
                ggml_cuda_moe_stream_q80_probe(
                        src0->type,
                        src0->name,
                        cur_a,
                        src0_cur,
                        ne01, ne00, nb01,
                        q80_base,
                        q80_row_size,
                        ne11,
                        cne1,
                        (const float *) dst->data,
                        nb1, nb2,
                        (const ggml_moe_stream_row_mapping *) (matrix_rows + cur_a * ids->ne[0] * ids->ne[1]));
            }
        }
        ggml_barrier(params->threadpool);
    }

    if (ggml_moe_stream_q80_hot_batch_probe_enabled() &&
            ggml_cuda_moe_stream_q80_hot_batch_probe &&
            src0->type == GGML_TYPE_MXFP4 &&
            src1->type != vec_dot_type &&
            vec_dot_type == GGML_TYPE_Q8_0 &&
            ne13 == 1 &&
            dst->type == GGML_TYPE_F32) {
        ggml_barrier(params->threadpool);
        if (ith == 0) {
            const void * q80_base = params->wdata;
            const size_t q80_row_size = ggml_row_size(vec_dot_type, ne10);
            ggml_cuda_moe_stream_q80_hot_batch_probe(
                    src0->type,
                    src0->name,
                    n_as,
                    ne01, ne00, nb01,
                    q80_base,
                    q80_row_size,
                    ne11,
                    (const float *) src1->data,
                    nb11,
                    nb12,
                    matrix_row_counts,
                    (const ggml_moe_stream_row_mapping *) matrix_rows,
                    ids->ne[0] * ids->ne[1],
                    (const float *) dst->data,
                    nb1, nb2);
        }
        ggml_barrier(params->threadpool);
    }

    if (ggml_moe_stream_q80_write_enabled() &&
            ggml_cuda_moe_stream_q80_write &&
            src0->type == GGML_TYPE_MXFP4 &&
            src1->type != vec_dot_type &&
            vec_dot_type == GGML_TYPE_Q8_0 &&
            ne13 == 1 &&
            dst->type == GGML_TYPE_F32) {
        ggml_barrier(params->threadpool);
        if (ith == 0) {
            const void * q80_base = params->wdata;
            const size_t q80_row_size = ggml_row_size(vec_dot_type, ne10);
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                const int64_t cne1 = matrix_row_counts[cur_a];
                if (cne1 == 0) {
                    continue;
                }
                const char * src0_cur = fallback_pack_mmap_ptrs[cur_a] ?
                    (const char *) fallback_pack_mmap_ptrs[cur_a] :
                    (const char *) src0->data + cur_a * nb02;
                ggml_cuda_moe_stream_q80_write(
                        src0->type,
                        src0->name,
                        cur_a,
                        src0_cur,
                        ne01, ne00, nb01,
                        q80_base,
                        q80_row_size,
                        ne11,
                        cne1,
                        (float *) dst->data,
                        nb1, nb2,
                        (const ggml_moe_stream_row_mapping *) (matrix_rows + cur_a * ids->ne[0] * ids->ne[1]));
            }
        }
        ggml_barrier(params->threadpool);
    }
    uint64_t t2= ggml_time_us();
    if(strstr(src0->name,"down") && t2-t0>2000){
        // printf("%ld:%ld ",t2-t0,t1-t0);
    }
}

static void ggml_compute_forward_mul_mat_id(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {
    if (ggml_model_type_is_kimi()) {
        const int64_t n_experts = dst->src[0]->ne[2];
        const size_t expert_bytes = (size_t) dst->src[0]->ne[1] * dst->src[0]->nb[1];
        if (ggml_cuda_moe_adapter_select_model_shape) {
            ggml_cuda_moe_adapter_select_model_shape(n_experts, expert_bytes);
        }
        if (ggml_kimi_uses_k3_local_adapter(n_experts, expert_bytes)) {
            ggml_compute_forward_mul_mat_id_kimi(params, dst);
        } else {
            ggml_compute_forward_mul_mat_id_common_kimi(params, dst);
        }
        return;
    }
    ggml_compute_forward_mul_mat_id_generic(params, dst);
}

static float ggml_moe_up_gate_activate(float x, enum ggml_unary_op op) {
    switch (op) {
        case GGML_UNARY_OP_SILU:
            return ggml_silu_f32(x);
        default:
            GGML_ABORT("unsupported MoE fused up/gate CPU activation");
    }
}

static float ggml_moe_up_gate_clamp(float x, float lo, float hi) {
    return MIN(hi, MAX(lo, x));
}

static float ggml_moe_up_gate_fuse_value(float up, float gate, enum ggml_unary_op op, float limit) {
    if (op == GGML_UNARY_OP_SILU && limit > 1.0e-6f) {
        const float gate_v = ggml_silu_f32(MIN(gate, limit));
        const float up_v = ggml_moe_up_gate_clamp(up, -limit, limit);
        return up_v * gate_v;
    }
    return up * ggml_moe_up_gate_activate(gate, op);
}

static void ggml_compute_forward_moe_up_gate_one_chunk(
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0_up,
    const struct ggml_tensor * src0_gate,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    const int64_t cur_a,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end,
    const char * src0_up_cur,
    const char * src0_gate_cur,
    const struct mmid_row_mapping * matrix_rows,
    const size_t row_size,
    const bool src1_cont,
    const void * wdata,
    enum ggml_unary_op op,
    float limit) {

    const enum ggml_type type_up = src0_up->type;
    const enum ggml_type type_gate = src0_gate->type;

    ggml_vec_dot_t const vec_dot_up      = type_traits_cpu[type_up].vec_dot;
    ggml_vec_dot_t const vec_dot_gate    = type_traits_cpu[type_gate].vec_dot;
    enum ggml_type const vec_dot_type_up = type_traits_cpu[type_up].vec_dot_type;
    enum ggml_type const vec_dot_type_gate = type_traits_cpu[type_gate].vec_dot_type;

    const int64_t ne00 = src0_up->ne[0];
    const int64_t ne11 = src1->ne[1];

    const size_t up_nb01 = src0_up->nb[1];
    const size_t gate_nb01 = src0_gate->nb[1];
    const size_t nb11 = src1->nb[1];
    const size_t nb12 = src1->nb[2];
    const size_t nb1  = dst->nb[1];
    const size_t nb2  = dst->nb[2];

    GGML_ASSERT(vec_dot_type_up == vec_dot_type_gate);
    const enum ggml_type vec_dot_type = vec_dot_type_up;

    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    float up_tmp[16];
    float gate_tmp[16];
    float fused_tmp[16];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);
                const int id = row_mapping.i1;

                const int64_t i11 = id % ne11;
                const int64_t i12 = row_mapping.i2;

                const char * src1_col = (const char *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11 + i12*ne11)*row_size
                    : (i11*nb11 + i12*nb12));

                float * dst_col = (float *) ((char *) dst->data + (id*nb1 + i12*nb2));

                const int64_t ir0_stop = MIN(iir0 + blck_0, ir0_end);
                for (int64_t ir0 = iir0; ir0 < ir0_stop; ++ir0) {
                    vec_dot_up(ne00,   &up_tmp[ir0 - iir0],   0, src0_up_cur   + ir0*up_nb01,   0, src1_col, 0, 1);
                    vec_dot_gate(ne00, &gate_tmp[ir0 - iir0], 0, src0_gate_cur + ir0*gate_nb01, 0, src1_col, 0, 1);
                }

                const int64_t n_cur = ir0_stop - iir0;
                if (op == GGML_UNARY_OP_SILU && limit > 1.0e-6f) {
                    for (int64_t i = 0; i < n_cur; ++i) {
                        gate_tmp[i] = MIN(gate_tmp[i], limit);
                        up_tmp[i]   = ggml_moe_up_gate_clamp(up_tmp[i], -limit, limit);
                    }
                    ggml_vec_swiglu_f32((int) n_cur, fused_tmp, gate_tmp, up_tmp);
                } else {
                    for (int64_t i = 0; i < n_cur; ++i) {
                        fused_tmp[i] = ggml_moe_up_gate_fuse_value(up_tmp[i], gate_tmp[i], op, limit);
                    }
                }

                memcpy(&dst_col[iir0], fused_tmp, n_cur*sizeof(float));
            }
        }
    }
}

static void ggml_compute_forward_moe_up_gate(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const int64_t n_experts = dst->src[0]->ne[2];
    const size_t expert_bytes = (size_t) dst->src[0]->ne[1] * dst->src[0]->nb[1];
    if (ggml_cuda_moe_adapter_select_model_shape) {
        ggml_cuda_moe_adapter_select_model_shape(n_experts, expert_bytes);
    }
    if (!ggml_kimi_uses_k3_local_adapter(n_experts, expert_bytes)) {
        ggml_compute_forward_moe_up_gate_common(params, dst);
        return;
    }

    const struct ggml_tensor * src0_up   = dst->src[0];
    const struct ggml_tensor * src0_gate = dst->src[1];
    const struct ggml_tensor * src1      = dst->src[2];
    const struct ggml_tensor * ids       = dst->src[3];

    const int ith = params->ith;
    const int nth = params->nth;
    const bool kimi_cpu_moe_profile = ggml_kimi_cpu_moe_profile_enabled();
    const uint64_t kimi_cpu_moe_total_start = (kimi_cpu_moe_profile && ith == 0) ? ggml_time_us() : 0;
    uint64_t kimi_cpu_moe_cuda_batch_this_us = 0;

    const enum ggml_type up_type = src0_up->type;
    const enum ggml_type gate_type = src0_gate->type;
    enum ggml_type    const vec_dot_type = type_traits_cpu[up_type].vec_dot_type;
    enum ggml_type    const gate_vec_dot_type = type_traits_cpu[gate_type].vec_dot_type;
    ggml_from_float_t const from_float   = type_traits_cpu[vec_dot_type].from_float;
    const bool same_weight_type = up_type == gate_type;
    const bool scoped_mixed_pair = ggml_kimi_moe_mixed_iq2_iq3_pair(up_type, gate_type);
    const float fused_limit = ggml_get_op_params_f32(dst, 1);

    GGML_ASSERT(src0_gate != NULL);
    GGML_ASSERT(same_weight_type || scoped_mixed_pair);
    GGML_ASSERT(vec_dot_type == gate_vec_dot_type);
    GGML_ASSERT(ggml_are_same_shape(src0_up, src0_gate));
    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == vec_dot_type);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_get_op_params_i32(dst, 0) == GGML_UNARY_OP_SILU);

    const int64_t ne00 = src0_up->ne[0];
    const int64_t ne01 = src0_up->ne[1];
    const int64_t n_as = src0_up->ne[2];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    const size_t nb01 = src0_up->nb[1];
    const size_t nb02 = src0_up->nb[2];
    const size_t gate_nb02 = src0_gate->nb[2];
    const size_t nb11 = src1->nb[1];
    const size_t nb12 = src1->nb[2];
    const size_t nb1  = dst->nb[1];
    const size_t nb2  = dst->nb[2];

    GGML_ASSERT(ne00 == ne10);
    GGML_ASSERT(ids->ne[0] % ne11 == 0);
    GGML_ASSERT(ids->ne[1] == ne12);
    GGML_ASSERT(ne13 == 1);

    const bool src1_cont = ggml_is_contiguous(src1);
    const int n_ids = ids->ne[0];

    void * wdata_cur = params->wdata;

    if (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_row_size(vec_dot_type, ggml_nelements(src1)), sizeof(int64_t));
    }

    int64_t * matrix_row_counts =
        incr_ptr_aligned(&wdata_cur, n_as*sizeof(int64_t), sizeof(int64_t));

    struct mmid_row_mapping * matrix_rows =
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping), sizeof(int64_t));

    char (*atomic_current_chunk)[CACHE_LINE_SIZE] =
        incr_ptr_aligned(&wdata_cur, CACHE_LINE_SIZE * n_as, CACHE_LINE_SIZE);

    GGML_ASSERT(params->wsize >= (size_t)((char *) wdata_cur - (char *) params->wdata));
    const uint64_t kimi_cpu_moe_convert_start = (kimi_cpu_moe_profile && ith == 0) ? ggml_time_us() : 0;
    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) src1->data + i13*src1->nb[3] + i12*nb12 + i11*nb11 + ne10_block_start*bs*src1->nb[0]),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
    }
    if (kimi_cpu_moe_profile && ith == 0) {
        ggml_kimi_cpu_moe_profile.up_gate.convert_us += ggml_time_us() - kimi_cpu_moe_convert_start;
    }

    if (ith == 0) {
        const uint64_t kimi_cpu_moe_route_start = kimi_cpu_moe_profile ? ggml_time_us() : 0;
        memset(matrix_row_counts, 0, n_as*sizeof(int64_t));

        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

                if (i02 == -1) {
                    memset((char *) dst->data + id * nb1 + iid1 * nb2, 0, (size_t) ne01 * sizeof(float));
                    continue;
                }
                assert(i02 >= 0 && i02 < n_as);

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (struct mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }
        if (kimi_cpu_moe_profile) {
            ggml_kimi_cpu_moe_profile.up_gate.route_us += ggml_time_us() - kimi_cpu_moe_route_start;
        }
        const size_t up_expert_bytes = (size_t)ne01 * nb01;
        const size_t gate_expert_bytes = (size_t)src0_gate->ne[1] * src0_gate->nb[1];
        ggml_ds4_grouped_retained_route_profile_record(
                src0_up->name,
                src0_up->type,
                src0_up->data,
                src0_up->nb[2],
                ids->ne[1] > 1,
                matrix_row_counts,
                n_as,
                up_expert_bytes);
        ggml_ds4_grouped_retained_route_profile_record(
                src0_gate->name,
                src0_gate->type,
                src0_gate->data,
                src0_gate->nb[2],
                ids->ne[1] > 1,
                matrix_row_counts,
                n_as,
                gate_expert_bytes);
        ggml_ds4_grouped_retained_route_profile_record_up_gate(
                src0_up->name,
                src0_gate->name,
                src0_up->type,
                src0_gate->type,
                src0_up->data,
                src0_up->nb[2],
                src0_gate->data,
                src0_gate->nb[2],
                ids->ne[1] > 1,
                matrix_row_counts,
                n_as,
                up_expert_bytes,
                gate_expert_bytes);
        ggml_ds4_grouped_retained_handoff_mark_up_gate(
                src0_up->name,
                src0_gate->name,
                dst->data,
                ne01,
                matrix_row_counts,
                n_as,
                ids->ne[0]*ids->ne[1]);
    }

    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    const uint64_t kimi_cpu_moe_route_barrier_start = (kimi_cpu_moe_profile && ith == 0) ? ggml_time_us() : 0;
    ggml_barrier(params->threadpool);
    if (kimi_cpu_moe_profile && ith == 0) {
        ggml_kimi_cpu_moe_profile.up_gate.route_barrier_us += ggml_time_us() - kimi_cpu_moe_route_barrier_start;
    }

    const bool use_gpu_stream =
        ggml_moe_gpu_provider_enabled(ids) &&
        ggml_cuda_moe_stream_up_gate_batch &&
        ggml_cuda_moe_stream_available &&
        ggml_cuda_moe_stream_available() &&
        ggml_cuda_moe_stream_supports_type(up_type) &&
        ggml_cuda_moe_stream_supports_type(gate_type) &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32;

    if (use_gpu_stream) {
        if (ith == 0) {
            const uint64_t kimi_cpu_moe_cuda_start = kimi_cpu_moe_profile ? ggml_time_us() : 0;
            if (ggml_moe_gate_batch_prefetch_enabled() &&
                    ggml_cuda_moe_stream_batch_preload_active_from_pack) {
                static bool same_layer_prefetch_logged = false;
                if (!same_layer_prefetch_logged) {
                    fprintf(stderr, "[moe_stream_cpu] same-layer prefetch before up/gate batch: gate=%s n_as=%" PRId64 " expert_bytes=%zu\n",
                            src0_gate->name, (int64_t) n_as,
                            (size_t) src0_gate->ne[1] * src0_gate->nb[1]);
                    same_layer_prefetch_logged = true;
                }
                const int same_layer_jobs = ggml_cuda_moe_stream_batch_preload_active_from_pack(
                        gate_type,
                        src0_gate->name,
                        n_as,
                        (size_t) src0_gate->ne[1] * src0_gate->nb[1],
                        matrix_row_counts);
                (void) same_layer_jobs;
            }
            const char * q4_colibri_pilot_prefetch_env = getenv("GGML_MOE_Q4_COLIBRI_PILOT_PREFETCH");
            if (q4_colibri_pilot_prefetch_env &&
                    q4_colibri_pilot_prefetch_env[0] &&
                    q4_colibri_pilot_prefetch_env[0] != '0' &&
                    ggml_cuda_moe_stream_q4_colibri_pilot_prefetch_from_gate) {
                const int pilot_prefetch_jobs = ggml_cuda_moe_stream_q4_colibri_pilot_prefetch_from_gate(
                        gate_type,
                        src0_gate->name,
                        n_as,
                        (size_t) src0_gate->ne[1] * src0_gate->nb[1],
                        matrix_row_counts);
                (void) pilot_prefetch_jobs;
            }
            const bool done = ggml_cuda_moe_stream_up_gate_batch(
                up_type,
                gate_type,
                src0_up->name, src0_up->data,
                src0_gate->name, src0_gate->data,
                n_as,
                ne01, ne00, nb01, nb02,
                (size_t) ne01 * nb01,
                src0_gate->nb[1], src0_gate->nb[2],
                (size_t) src0_gate->ne[1] * src0_gate->nb[1],
                (const float *) src1->data,
                nb11, nb12,
                (float *) dst->data,
                nb1, nb2,
                ggml_get_op_params_i32(dst, 0),
                fused_limit,
                matrix_row_counts,
                (const ggml_moe_stream_row_mapping *) matrix_rows,
                ids->ne[0]*ids->ne[1]);
            if (kimi_cpu_moe_profile) {
                const uint64_t elapsed_us = ggml_time_us() - kimi_cpu_moe_cuda_start;
                ggml_kimi_cpu_moe_profile.up_gate.cuda_batch_us += elapsed_us;
                kimi_cpu_moe_cuda_batch_this_us += elapsed_us;
                if (done) {
                    ggml_kimi_cpu_moe_profile.up_gate.cuda_batch_accepted++;
                } else {
                    ggml_kimi_cpu_moe_profile.up_gate.cuda_batch_declined++;
                }
            }
            if (done) {
                memset(matrix_row_counts, 0, n_as*sizeof(int64_t));
            }
        }

        const uint64_t kimi_cpu_moe_post_cuda_barrier_start = (kimi_cpu_moe_profile && ith == 0) ? ggml_time_us() : 0;
        ggml_barrier(params->threadpool);
        if (kimi_cpu_moe_profile && ith == 0) {
            ggml_kimi_cpu_moe_profile.up_gate.post_cuda_barrier_us += ggml_time_us() - kimi_cpu_moe_post_cuda_barrier_start;
        }
    }

    const size_t row_size = ggml_row_size(vec_dot_type, ne10);
    const enum ggml_unary_op op = (enum ggml_unary_op) ggml_get_op_params_i32(dst, 0);

    const bool moe_fallback_reason_profile = ggml_moe_fallback_reason_profile_enabled();
    const uint64_t moe_fallback_reason_start = (moe_fallback_reason_profile && ith == 0) ? ggml_time_us() : 0;
    const uint64_t kimi_cpu_moe_fallback_start = (kimi_cpu_moe_profile && ith == 0) ? ggml_time_us() : 0;
    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = matrix_row_counts[cur_a];

        if (cne1 == 0) {
            continue;
        }

        const char * src0_up_cur   = (const char *) src0_up->data   + cur_a * nb02;
        const char * src0_gate_cur = (const char *) src0_gate->data + cur_a * gate_nb02;
        const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;

        const int64_t nr0 = ne01;
        const int64_t nr1 = cne1;

        int chunk_size = 16;
        if (nr0 == 1 || nr1 == 1) {
            chunk_size = 64;
        }

        const bool disable_chunking = ggml_is_numa();

        int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

        if (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        int current_chunk = ith;
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        while (current_chunk < nchunk0 * nchunk1) {
            const int64_t ith0 = current_chunk % nchunk0;
            const int64_t ith1 = current_chunk / nchunk0;

            const int64_t ir0_start = dr0 * ith0;
            const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

            const int64_t ir1_start = dr1 * ith1;
            const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

            const bool trace_cpu_chunk = ggml_moe_cpu_chunk_trace_enabled();
            const double trace_t0_ms = trace_cpu_chunk ? ggml_moe_cpu_trace_now_ms() : 0.0;

            ggml_compute_forward_moe_up_gate_one_chunk(
                dst, src0_up, src0_gate, src1, ids, cur_a,
                ir0_start, ir0_end, ir1_start, ir1_end,
                src0_up_cur, src0_gate_cur, matrix_rows, row_size, src1_cont, wdata, op, fused_limit);

            if (trace_cpu_chunk) {
                const double trace_t1_ms = ggml_moe_cpu_trace_now_ms();
                ggml_moe_cpu_chunk_trace_write(
                    "up_gate",
                    src0_up->name,
                    src0_up->type,
                    cur_a,
                    ith,
                    nth,
                    cne1,
                    ir0_start,
                    ir0_end,
                    ir1_start,
                    ir1_end,
                    (size_t) ne01 * nb01 + (size_t) src0_gate->ne[1] * src0_gate->nb[1],
                    trace_t1_ms - trace_t0_ms);
            }

            if (nth >= nchunk0 * nchunk1) {
                break;
            }

            current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }
    if (moe_fallback_reason_profile && ith == 0) {
        const uint64_t moe_fallback_reason_us = ggml_time_us() - moe_fallback_reason_start;
        int64_t fallback_rows = 0;
        for (int cur_a = 0; cur_a < n_as; ++cur_a) {
            fallback_rows += matrix_row_counts[cur_a];
        }
        const char * batch_reason_name = use_gpu_stream ? "eligible" : "upgate_batch_precondition_failed";
        const char * final_reason_name = use_gpu_stream ? "upgate_batch_declined_internal" : "upgate_batch_not_attempted";
        if (fallback_rows > 0) {
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                const int64_t cne1 = matrix_row_counts[cur_a];
                if (cne1 == 0) {
                    continue;
                }
                const uint64_t expert_fallback_us =
                    (uint64_t) (((double) moe_fallback_reason_us * (double) cne1) / (double) fallback_rows);
                ggml_moe_fallback_reason_profile_record(
                        "up_gate",
                        src0_up->name,
                        src0_up->type,
                        ids->ne[1] > 1,
                        cur_a,
                        cne1,
                        (size_t) ne01 * nb01 + (size_t) src0_gate->ne[1] * src0_gate->nb[1],
                        expert_fallback_us,
                        batch_reason_name,
                        "not_applicable",
                        final_reason_name,
                        use_gpu_stream,
                        false,
                        false,
                        false);
                ggml_moe_fallback_source_probe_record(
                        "up_gate",
                        src0_up->name,
                        ids->ne[1] > 1,
                        cur_a,
                        cne1,
                        expert_fallback_us,
                        batch_reason_name,
                        "not_applicable",
                        final_reason_name,
                        src0_up,
                        src1,
                        ids,
                        dst);
            }
        }
    }
    if (kimi_cpu_moe_profile && ith == 0) {
        const uint64_t now_us = ggml_time_us();
        const uint64_t fallback_us = now_us - kimi_cpu_moe_fallback_start;
        const uint64_t total_us = now_us - kimi_cpu_moe_total_start;
        ggml_kimi_cpu_moe_profile.up_gate.fallback_us += fallback_us;
        ggml_kimi_cpu_moe_profile.up_gate.total_us += total_us;
        ggml_kimi_cpu_moe_profile.up_gate.calls++;
        ggml_kimi_cpu_moe_profile_record_phase(
                &ggml_kimi_cpu_moe_profile.up_gate,
                ids->ne[1] > 1,
                total_us,
                kimi_cpu_moe_cuda_batch_this_us,
                0,
                fallback_us);
    }
}

/////////////////////////////////

static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    GGML_ASSERT(params);

    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        return;
    }

    // extra_buffer op?
    if (ggml_cpu_extra_compute_forward(params, tensor)) {
        return;
    }

    switch (tensor->op) {
        case GGML_OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor);
            } break;
        case GGML_OP_ADD:
            {
                ggml_compute_forward_add(params, tensor);
            } break;
        case GGML_OP_ADD_ID:
            {
                ggml_compute_forward_add_id(params, tensor);
            } break;
        case GGML_OP_ADD1:
            {
                ggml_compute_forward_add1(params, tensor);
            } break;
        case GGML_OP_ACC:
            {
                ggml_compute_forward_acc(params, tensor);
            } break;
        case GGML_OP_SUB:
            {
                ggml_compute_forward_sub(params, tensor);
            } break;
        case GGML_OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor);
            } break;
        case GGML_OP_DIV:
            {
                ggml_compute_forward_div(params, tensor);
            } break;
        case GGML_OP_SQR:
            {
                ggml_compute_forward_sqr(params, tensor);
            } break;
        case GGML_OP_SQRT:
            {
                ggml_compute_forward_sqrt(params, tensor);
            } break;
        case GGML_OP_LOG:
            {
                ggml_compute_forward_log(params, tensor);
            } break;
        case GGML_OP_SIN:
            {
                ggml_compute_forward_sin(params, tensor);
            } break;
        case GGML_OP_COS:
            {
                ggml_compute_forward_cos(params, tensor);
            } break;
        case GGML_OP_SUM:
            {
                ggml_compute_forward_sum(params, tensor);
            } break;
        case GGML_OP_SUM_ROWS:
            {
                ggml_compute_forward_sum_rows(params, tensor);
            } break;
        case GGML_OP_CUMSUM:
            {
                ggml_compute_forward_cumsum(params, tensor);
            } break;
        case GGML_OP_MEAN:
            {
                ggml_compute_forward_mean(params, tensor);
            } break;
        case GGML_OP_ARGMAX:
            {
                ggml_compute_forward_argmax(params, tensor);
            } break;
        case GGML_OP_COUNT_EQUAL:
            {
                ggml_compute_forward_count_equal(params, tensor);
            } break;
        case GGML_OP_REPEAT:
            {
                ggml_compute_forward_repeat(params, tensor);
            } break;
        case GGML_OP_REPEAT_BACK:
            {
                ggml_compute_forward_repeat_back(params, tensor);
            } break;
        case GGML_OP_CONCAT:
            {
                ggml_compute_forward_concat(params, tensor);
            } break;
        case GGML_OP_SILU_BACK:
            {
                ggml_compute_forward_silu_back(params, tensor);
            } break;
        case GGML_OP_NORM:
            {
                ggml_compute_forward_norm(params, tensor);
            } break;
        case GGML_OP_RMS_NORM:
            {
                ggml_compute_forward_rms_norm(params, tensor);
            } break;
        case GGML_OP_RMS_NORM_BACK:
            {
                ggml_compute_forward_rms_norm_back(params, tensor);
            } break;
        case GGML_OP_GROUP_NORM:
            {
                ggml_compute_forward_group_norm(params, tensor);
            } break;
        case GGML_OP_L2_NORM:
            {
                ggml_compute_forward_l2_norm(params, tensor);
            } break;
        case GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                ggml_compute_forward_mul_mat_id(params, tensor);
            } break;
        case GGML_OP_MOE_FUSED_UP_GATE:
            {
                ggml_compute_forward_moe_up_gate(params, tensor);
            } break;
        case GGML_OP_HC_WEIGHTED_SUM:
            {
                ggml_compute_forward_hc_weighted_sum(params, tensor);
            } break;
        case GGML_OP_DSV4_HC_COMB:
            {
                ggml_compute_forward_dsv4_hc_comb(params, tensor);
            } break;
        case GGML_OP_DSV4_HC_POST:
            {
                ggml_compute_forward_dsv4_hc_post(params, tensor);
            } break;
        case GGML_OP_OUT_PROD:
            {
                ggml_compute_forward_out_prod(params, tensor);
            } break;
        case GGML_OP_SCALE:
            {
                ggml_compute_forward_scale(params, tensor);
            } break;
        case GGML_OP_SET:
            {
                ggml_compute_forward_set(params, tensor);
            } break;
        case GGML_OP_CPY:
            {
                ggml_compute_forward_cpy(params, tensor);
            } break;        case GGML_OP_CONT:
            {
                ggml_compute_forward_cont(params, tensor);
            } break;
        case GGML_OP_GET_ROWS:
            {
                ggml_compute_forward_get_rows(params, tensor);
            } break;
        case GGML_OP_GET_ROWS_BACK:
            {
                ggml_compute_forward_get_rows_back(params, tensor);
            } break;
        case GGML_OP_SET_ROWS:
            {
                ggml_compute_forward_set_rows(params, tensor);
            } break;
        case GGML_OP_DIAG:
            {
                ggml_compute_forward_diag(params, tensor);
            } break;
        case GGML_OP_DIAG_MASK_INF:
            {
                ggml_compute_forward_diag_mask_inf(params, tensor);
            } break;
        case GGML_OP_DIAG_MASK_ZERO:
            {
                ggml_compute_forward_diag_mask_zero(params, tensor);
            } break;
        case GGML_OP_ROW_MASK:
            {
                ggml_compute_forward_row_mask(params, tensor);
            } break;
        case GGML_OP_SOFT_MAX:
            {
                ggml_compute_forward_soft_max(params, tensor);
            } break;
        case GGML_OP_SOFT_MAX_BACK:
            {
                ggml_compute_forward_soft_max_ext_back(params, tensor);
            } break;
        case GGML_OP_ROPE:
            {
                ggml_compute_forward_rope(params, tensor);
            } break;
        case GGML_OP_ROPE_BACK:
            {
                ggml_compute_forward_rope_back(params, tensor);
            } break;
        case GGML_OP_CLAMP:
            {
                ggml_compute_forward_clamp(params, tensor);
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_compute_forward_conv_transpose_1d(params, tensor);
            } break;
        case GGML_OP_IM2COL:
            {
                ggml_compute_forward_im2col(params, tensor);
            } break;
        case GGML_OP_IM2COL_BACK:
            {
                ggml_compute_forward_im2col_back_f32(params, tensor);
            } break;
        case GGML_OP_IM2COL_3D:
            {
                ggml_compute_forward_im2col_3d(params, tensor);
            } break;
        case GGML_OP_CONV_2D:
            {
                ggml_compute_forward_conv_2d(params, tensor);
            } break;
        case GGML_OP_CONV_3D:
            {
                ggml_compute_forward_conv_3d(params, tensor);
            } break;
        case GGML_OP_CONV_2D_DW:
            {
                ggml_compute_forward_conv_2d_dw(params, tensor);
            } break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                ggml_compute_forward_conv_transpose_2d(params, tensor);
            } break;
        case GGML_OP_POOL_1D:
            {
                ggml_compute_forward_pool_1d(params, tensor);
            } break;
        case GGML_OP_POOL_2D:
            {
                ggml_compute_forward_pool_2d(params, tensor);
            } break;
        case GGML_OP_POOL_2D_BACK:
            {
                ggml_compute_forward_pool_2d_back(params, tensor);
            } break;
        case GGML_OP_UPSCALE:
            {
                ggml_compute_forward_upscale(params, tensor);
            } break;
        case GGML_OP_PAD:
            {
                ggml_compute_forward_pad(params, tensor);
            } break;
        case GGML_OP_PAD_REFLECT_1D:
            {
                ggml_compute_forward_pad_reflect_1d(params, tensor);
            } break;
        case GGML_OP_ROLL:
            {
                ggml_compute_forward_roll(params, tensor);
            } break;
        case GGML_OP_ARANGE:
            {
                ggml_compute_forward_arange(params, tensor);
            } break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            {
                ggml_compute_forward_timestep_embedding(params, tensor);
            } break;
        case GGML_OP_ARGSORT:
            {
                ggml_compute_forward_argsort(params, tensor);
            } break;
        case GGML_OP_TOP_K:
            {
                ggml_compute_forward_top_k(params, tensor);
            } break;
        case GGML_OP_LEAKY_RELU:
            {
                ggml_compute_forward_leaky_relu(params, tensor);
            } break;
        case GGML_OP_TRI:
            {
                ggml_compute_forward_tri(params, tensor);
            } break;
        case GGML_OP_FILL:
            {
                ggml_compute_forward_fill(params, tensor);
            } break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_compute_forward_flash_attn_ext(params, tensor);
            } break;
        case GGML_OP_FLASH_ATTN_BACK:
            {
                int32_t t = ggml_get_op_params_i32(tensor, 0);
                GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                ggml_compute_forward_flash_attn_back(params, masked, tensor);
            } break;
        case GGML_OP_SSM_CONV:
            {
                ggml_compute_forward_ssm_conv(params, tensor);
            } break;
        case GGML_OP_SSM_SCAN:
            {
                ggml_compute_forward_ssm_scan(params, tensor);
            } break;
        case GGML_OP_WIN_PART:
            {
                ggml_compute_forward_win_part(params, tensor);
            } break;
        case GGML_OP_WIN_UNPART:
            {
                ggml_compute_forward_win_unpart(params, tensor);
            } break;
        case GGML_OP_UNARY:
            {
                ggml_compute_forward_unary(params, tensor);
            } break;
        case GGML_OP_GLU:
            {
                ggml_compute_forward_glu(params, tensor);
                ggml_ds4_grouped_retained_handoff_mark_glu_act(tensor);
            } break;
        case GGML_OP_GET_REL_POS:
            {
                ggml_compute_forward_get_rel_pos(params, tensor);
            } break;
        case GGML_OP_ADD_REL_POS:
            {
                ggml_compute_forward_add_rel_pos(params, tensor);
            } break;
        case GGML_OP_RWKV_WKV6:
            {
                ggml_compute_forward_rwkv_wkv6(params, tensor);
            } break;
        case GGML_OP_GATED_LINEAR_ATTN:
            {
                ggml_compute_forward_gla(params, tensor);
            } break;
        case GGML_OP_RWKV_WKV7:
            {
                ggml_compute_forward_rwkv_wkv7(params, tensor);
            } break;
        case GGML_OP_SOLVE_TRI:
            {
                ggml_compute_forward_solve_tri(params, tensor);
            } break;
        case GGML_OP_GATED_DELTA_NET:
            {
                ggml_compute_forward_gated_delta_net(params, tensor);
            } break;
        case GGML_OP_MAP_CUSTOM1:
            {
                ggml_compute_forward_map_custom1(params, tensor);
            }
            break;
        case GGML_OP_MAP_CUSTOM2:
            {
                ggml_compute_forward_map_custom2(params, tensor);
            }
            break;
        case GGML_OP_MAP_CUSTOM3:
            {
                ggml_compute_forward_map_custom3(params, tensor);
            }
            break;
        case GGML_OP_CUSTOM:
            {
                ggml_compute_forward_custom(params, tensor);
            }
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            {
                ggml_compute_forward_cross_entropy_loss(params, tensor);
            }
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            {
                ggml_compute_forward_cross_entropy_loss_back(params, tensor);
            }
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            {
                ggml_compute_forward_opt_step_adamw(params, tensor);
            }
            break;
        case GGML_OP_OPT_STEP_SGD:
            {
                ggml_compute_forward_opt_step_sgd(params, tensor);
            }
            break;
        case GGML_OP_NONE:
            {
                // nop
            } break;
        case GGML_OP_RESHAPE:
            {
                // nop
            } break;
        case GGML_OP_PERMUTE:
            {
                // nop
            } break;
        case GGML_OP_VIEW:
            {
                // nop
            } break;
        case GGML_OP_TRANSPOSE:
            {
                // nop
            } break;
        case GGML_OP_COUNT:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Android's libc implementation "bionic" does not support setting affinity
#if defined(__gnu_linux__)
static void set_numa_thread_affinity(int thread_n) {
    if (!ggml_is_numa()) {
        return;
    }

    int node_num;
    int rv;
    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    switch(g_state.numa.numa_strategy) {
        case GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            node_num = thread_n % g_state.numa.n_nodes;
            break;
        case GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            node_num = g_state.numa.current_node;
            break;
        case GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            if (rv) {
                fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
            }
            return;
        default:
            return;
    }

    struct ggml_numa_node * node = &g_state.numa.nodes[node_num];

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (size_t i = 0; i < node->n_cpus; ++i) {
        CPU_SET_S(node->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

static void clear_numa_thread_affinity(void) {
    if (!ggml_is_numa()) {
        return;
    }

    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (unsigned i = 0; i < g_state.numa.total_cpus; ++i) {
        CPU_SET_S(i, setsize, cpus);
    }

    int rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}
#else
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
static void set_numa_thread_affinity(int thread_n) { UNUSED(thread_n);  }
static void clear_numa_thread_affinity(void) {}
#endif

static int ggml_get_n_tasks(struct ggml_tensor * node, int n_threads) {
    int n_tasks = 0;

    if (ggml_is_empty(node)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        return n_tasks;
    }

    switch (node->op) {
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_ACC:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_FILL:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_SUB:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_ARGMAX:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_COUNT_EQUAL:
        case GGML_OP_SOLVE_TRI:
        case GGML_OP_GATED_DELTA_NET:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_REPEAT:
        case GGML_OP_REPEAT_BACK:
        case GGML_OP_LEAKY_RELU:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(node)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                case GGML_UNARY_OP_FP4_ACT_QUANT:
                case GGML_UNARY_OP_FP8_ACT_QUANT:
                    {
                        n_tasks = n_threads;
                    } break;
                case GGML_UNARY_OP_SINKHORN_4X4:
                    {
                        n_tasks = 1;
                    } break;

                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_XIELU:
                    {
                        n_tasks = n_threads;
                    } break;
                default:
                    GGML_ABORT("fatal error");
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(node)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    {
                        n_tasks = n_threads;
                    } break;
                default:
                    GGML_ABORT("fatal error");
            }
            break;
        case GGML_OP_SILU_BACK:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_RMS_NORM_BACK:
        case GGML_OP_L2_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_CONCAT:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_MOE_FUSED_UP_GATE:
        case GGML_OP_HC_WEIGHTED_SUM:
        case GGML_OP_DSV4_HC_COMB:
        case GGML_OP_DSV4_HC_POST:
        case GGML_OP_OUT_PROD:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_GET_ROWS:
        case GGML_OP_SET_ROWS:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                //n_tasks = n_threads;
                n_tasks = 1;
            } break;
        case GGML_OP_SCALE:
        case GGML_OP_SET:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_GET_ROWS_BACK:
        case GGML_OP_DIAG:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_DIAG_MASK_ZERO:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_ROW_MASK:
        case GGML_OP_SOFT_MAX_BACK:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_ADD_REL_POS:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_CLAMP:
            {
                n_tasks = 1; //TODO
            } break;
        case GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_threads, ggml_nrows(node->src[0]));
            } break;
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_BACK:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_3D:
        case GGML_OP_CONV_2D_DW:
        case GGML_OP_CONV_TRANSPOSE_1D:
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
        case GGML_OP_POOL_2D_BACK:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ROLL:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_FLASH_ATTN_BACK:
        case GGML_OP_SSM_CONV:
        case GGML_OP_SSM_SCAN:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            {
                const int64_t n_heads = node->src[1]->ne[1];
                n_tasks = MIN(n_threads, n_heads);
            } break;
        case GGML_OP_WIN_PART:
        case GGML_OP_WIN_UNPART:
        case GGML_OP_GET_REL_POS:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_MAP_CUSTOM1:
            {
                struct ggml_map_custom1_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_MAP_CUSTOM2:
            {
                struct ggml_map_custom2_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_MAP_CUSTOM3:
            {
                struct ggml_map_custom3_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_CUSTOM:
            {
                struct ggml_custom_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_NONE:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_COUNT:
            {
                GGML_ABORT("fatal error");
            }
        default:
            {
                fprintf(stderr, "%s: op not implemented: ", __func__);
                if (node->op < GGML_OP_COUNT) {
                    fprintf(stderr, "%s\n", ggml_op_name(node->op));
                } else {
                    fprintf(stderr, "%d\n", node->op);
                }
                GGML_ABORT("fatal error");
            }
    }

    assert(n_tasks > 0);

    return n_tasks;
}

static thread_ret_t ggml_graph_compute_secondary_thread(void* data);

#if defined(_WIN32)
#include "windows.h"

// TODO: support > 64 CPUs
static bool ggml_thread_apply_affinity(bool * mask) {
    HANDLE    h = GetCurrentThread();
    uint64_t  bitmask = 0ULL;

    assert(GGML_MAX_N_THREADS >= 64);

    for (int32_t i = 0; i < 8; i++) {
        int32_t idx = i * 8;
        uint8_t val = 0;
        val |= mask[idx + 0] << 0;
        val |= mask[idx + 1] << 1;
        val |= mask[idx + 2] << 2;
        val |= mask[idx + 3] << 3;
        val |= mask[idx + 4] << 4;
        val |= mask[idx + 5] << 5;
        val |= mask[idx + 6] << 6;
        val |= mask[idx + 7] << 7;
        bitmask |= (uint64_t)val << idx;
    }

    for (int32_t i = 64; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) {
            fprintf(stderr, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
            break;
        }
    }

    DWORD_PTR m = (DWORD_PTR)bitmask;

    m = SetThreadAffinityMask(h, m);

    return m != 0;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_PRIORITY_NORMAL;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      p = THREAD_PRIORITY_BELOW_NORMAL;  break;
        case GGML_SCHED_PRIO_NORMAL:   p = THREAD_PRIORITY_NORMAL;        break;
        case GGML_SCHED_PRIO_MEDIUM:   p = THREAD_PRIORITY_ABOVE_NORMAL;  break;
        case GGML_SCHED_PRIO_HIGH:     p = THREAD_PRIORITY_HIGHEST;       break;
        case GGML_SCHED_PRIO_REALTIME: p = THREAD_PRIORITY_TIME_CRITICAL; break;
    }

    if (prio != GGML_SCHED_PRIO_LOW) {
        // Tell Windows that this thread should not be throttled (needs its own CPU core).
        // Newer Windows 11 versions aggressively park (offline) CPU cores and often place
        // all our threads onto the first 4 cores which results in terrible performance with
        // n_threads > 4
        #if _WIN32_WINNT >= 0x0602
        THREAD_POWER_THROTTLING_STATE t;
        ZeroMemory(&t, sizeof(t));
        t.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
        t.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        t.StateMask   = 0;

        if (!SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &t, sizeof(t))) {
            GGML_LOG_DEBUG("failed to disable thread power throttling %d : (%d)\n", prio, (int) GetLastError());
            return false;
        }
        #endif
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    if (!SetThreadPriority(GetCurrentThread(), p)) {
        fprintf(stderr, "warn: failed to set thread priority %d : (%d)\n", prio, (int) GetLastError());
        return false;
    }

    return true;
}

#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/resource.h>

static bool ggml_thread_apply_affinity(const bool * mask) {
    // Not supported on Apple platforms
    UNUSED(mask);
    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        // TODO: there seems to be no way to set lower prio on Apple platforms
        case GGML_SCHED_PRIO_LOW:      policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#elif defined(__gnu_linux__)
// TODO: this may not work on BSD, to be verified

static bool ggml_thread_apply_affinity(const bool * mask) {
    cpu_set_t cpuset;
    int err;

    CPU_ZERO(&cpuset);

    for (uint32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) {
            GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), i);
            CPU_SET(i, &cpuset);
        }
    }

#ifdef __ANDROID__
    err = sched_setaffinity(0, sizeof(cpuset), &cpuset);
    if (err < 0) {
        err = errno;
    }
#else
    err = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif
    if (err != 0) {
        fprintf(stderr, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (unsigned long long)mask, strerror(err), err);
        return false;
    }

    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      policy = SCHED_BATCH; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#else // unsupported platforms

static bool ggml_thread_apply_affinity(const bool * mask) {
    UNUSED(mask);
    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    UNUSED(prio);
    return true;
}

#endif

static bool ggml_thread_cpumask_is_valid(const bool * mask) {
    for (int i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) { return true; }
    }
    return false;
}

static void ggml_thread_cpumask_next(const bool * global_mask, bool * local_mask, bool strict, int32_t* iter) {
    if (!strict) {
        memcpy(local_mask, global_mask, GGML_MAX_N_THREADS);
        return;
    } else {
        memset(local_mask, 0, GGML_MAX_N_THREADS);
        int32_t base_idx = *iter;
        for (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
            int32_t idx = base_idx + i;
            if (idx >= GGML_MAX_N_THREADS) {
                // Just a cheaper modulo
                idx -= GGML_MAX_N_THREADS;
            }
            if (global_mask[idx]) {
                local_mask[idx] = 1;
                *iter = idx + 1;
                return;
            }
        }
    }
}

void ggml_threadpool_free(struct ggml_threadpool* threadpool) {
    if (!threadpool) return;

    const int n_threads = threadpool->n_threads;

#ifndef GGML_USE_OPENMP
    struct ggml_compute_state* workers = threadpool->workers;

    ggml_mutex_lock(&threadpool->mutex);

    threadpool->stop = true;
    threadpool->pause = false;

    ggml_cond_broadcast(&threadpool->cond);
    ggml_mutex_unlock(&threadpool->mutex);

    for (int j = 1; j < n_threads; j++) {
        int32_t rc = ggml_thread_join(workers[j].thrd, NULL);
        GGML_ASSERT(rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED);
        UNUSED(rc);
    }

    ggml_mutex_destroy(&threadpool->mutex);
    ggml_cond_destroy(&threadpool->cond);
#endif // GGML_USE_OPENMP

    const size_t workers_size = sizeof(struct ggml_compute_state) * n_threads;
    ggml_aligned_free(threadpool->workers, workers_size);
    ggml_aligned_free(threadpool, sizeof(struct ggml_threadpool));
}

#ifndef GGML_USE_OPENMP
// pause/resume must be called under mutex
static void ggml_threadpool_pause_locked(struct ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->pause = true;
    ggml_cond_broadcast(&threadpool->cond);
}

static void ggml_threadpool_resume_locked(struct ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->pause = false;
    ggml_cond_broadcast(&threadpool->cond);
}
#endif

void ggml_threadpool_pause(struct ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    if (!threadpool->pause) {
       ggml_threadpool_pause_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#else
    UNUSED(threadpool);
#endif
}

void ggml_threadpool_resume(struct ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    if (threadpool->pause) {
       ggml_threadpool_resume_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#else
    UNUSED(threadpool);
#endif
}

struct ggml_cplan ggml_graph_plan(
          const struct ggml_cgraph * cgraph,
                               int   n_threads,
            struct ggml_threadpool * threadpool) {

    if (threadpool == NULL) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    if (n_threads <= 0) {
        n_threads = threadpool ? threadpool->n_threads : GGML_DEFAULT_N_THREADS;
    }

#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    // Emscripten without pthreads support can only use a single thread
    n_threads = 1;
#endif

    size_t work_size = 0;

    struct ggml_cplan cplan;
    memset(&cplan, 0, sizeof(struct ggml_cplan));

    int max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        const int n_tasks = ggml_get_n_tasks(node, n_threads);

        max_tasks = MAX(max_tasks, n_tasks);

        size_t cur = 0;

        if (!ggml_cpu_extra_work_size(n_threads, node, &cur)) {
            switch (node->op) {
                case GGML_OP_CPY:
                case GGML_OP_DUP:
                    {
                        if (ggml_is_quantized(node->type) ||
                            // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                            (node->src[0]->type == GGML_TYPE_F16  && node->src[1] && node->src[1]->type == GGML_TYPE_BF16) ||
                            (node->src[0]->type == GGML_TYPE_BF16 && node->src[1] && node->src[1]->type == GGML_TYPE_F16) ||
                            // conversion between F32 and I32
                            (node->src[0]->type == GGML_TYPE_F32 && node->src[1] && node->src[1]->type == GGML_TYPE_I32) ||
                            (node->src[0]->type == GGML_TYPE_I32 && node->src[1] && node->src[1]->type == GGML_TYPE_F32)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_ADD:
                case GGML_OP_ADD_ID:
                case GGML_OP_ADD1:
                    {
                        if (ggml_is_quantized(node->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_ACC:
                    {
                        if (ggml_is_quantized(node->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[1]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_COUNT_EQUAL:
                    {
                        cur = ggml_type_size(node->type)*n_tasks;
                    } break;
                case GGML_OP_MUL_MAT:
                    {
                        const enum ggml_type vec_dot_type = type_traits_cpu[node->src[0]->type].vec_dot_type;

                        if (node->src[1]->type != vec_dot_type) {
                            cur = ggml_row_size(vec_dot_type, ggml_nelements(node->src[1]));
                        }
                    } break;
                case GGML_OP_MUL_MAT_ID:
                    {
                        cur = 0;
                        const struct ggml_tensor * src0 = node->src[0];
                        const struct ggml_tensor * src1 = node->src[1];
                        const struct ggml_tensor * ids = node->src[2];
                        const enum ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
                        const int n_as = src0->ne[2];
                        // src1
                        if (src1->type != vec_dot_type) {
                            cur += ggml_row_size(vec_dot_type, ggml_nelements(src1)) + sizeof(int64_t);
                        }
                        // matrix_row_counts
                        cur += n_as * sizeof(int64_t) + sizeof(int64_t);
                        // matrix_rows
                        cur += n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping) + sizeof(int64_t);
                        // atomic_current_chunk
                        cur += CACHE_LINE_SIZE*n_as + CACHE_LINE_SIZE;
                        // CPU fallback expert-pack mmap pointers
                        cur += n_as * sizeof(void *) + sizeof(void *);
                        // CPU fallback source-touch profile timings
                        cur += n_as * sizeof(uint64_t) + sizeof(uint64_t);
                        // GPU top-k / CPU-tail split counts
                        cur += n_as * sizeof(int64_t) + sizeof(int64_t);
                        // GPU top-k / CPU-tail split rows
                        cur += n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping) + sizeof(int64_t);
                    } break;
                case GGML_OP_MOE_FUSED_UP_GATE:
                    {
                        cur = 0;
                        const struct ggml_tensor * src0 = node->src[0];
                        const struct ggml_tensor * src1 = node->src[2];
                        const struct ggml_tensor * ids = node->src[3];
                        const enum ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
                        const int n_as = src0->ne[2];
                        if (src1->type != vec_dot_type) {
                            cur += ggml_row_size(vec_dot_type, ggml_nelements(src1)) + sizeof(int64_t);
                        }
                        cur += n_as * sizeof(int64_t) + sizeof(int64_t);
                        cur += n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping) + sizeof(int64_t);
                        cur += CACHE_LINE_SIZE*n_as + CACHE_LINE_SIZE;
                        // The restored common adapter partitions prompt rows between
                        // GPU and CPU.  Only the validated Kimi-K3 expert-count and
                        // role-slice shape has its own local workspace layout.
                        const size_t expert_bytes = (size_t) src0->ne[1] * src0->nb[1];
                        if (!ggml_kimi_uses_k3_local_adapter(n_as, expert_bytes)) {
                            cur += n_as * sizeof(int64_t) + sizeof(int64_t);
                            cur += n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping) + sizeof(int64_t);
                        }
                    } break;
                case GGML_OP_OUT_PROD:
                    {
                        if (ggml_is_quantized(node->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_SET_ROWS:
                    {
                        if (node->src[0]->type == GGML_TYPE_F16 && node->type != GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_SOFT_MAX:
                case GGML_OP_ROPE:
                case GGML_OP_ROPE_BACK:
                    {
                        cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
                    } break;
                case GGML_OP_CONV_TRANSPOSE_1D:
                    {
                        GGML_ASSERT(node->src[0]->ne[3] == 1);
                        GGML_ASSERT(node->src[1]->ne[2] == 1);
                        GGML_ASSERT(node->src[1]->ne[3] == 1);

                        const int64_t ne00 = node->src[0]->ne[0];  // K
                        const int64_t ne01 = node->src[0]->ne[1];  // Cout
                        const int64_t ne02 = node->src[0]->ne[2];  // Cin
                        const int64_t ne10 = node->src[1]->ne[0];  // L
                        const int64_t ne11 = node->src[1]->ne[1];  // Cin

                        if ((node->src[0]->type == GGML_TYPE_F16 ||
                             node->src[0]->type == GGML_TYPE_BF16) &&
                            node->src[1]->type == GGML_TYPE_F32) {
                            cur += sizeof(ggml_fp16_t)*ne00*ne01*ne02;
                            cur += sizeof(ggml_fp16_t)*ne10*ne11;
                        } else if (node->src[0]->type == GGML_TYPE_F32 &&
                                   node->src[1]->type == GGML_TYPE_F32) {
                            cur += sizeof(float)*ne00*ne01*ne02;
                            cur += sizeof(float)*ne10*ne11;
                        } else {
                            GGML_ABORT("fatal error");
                        }
                    } break;
                case GGML_OP_CONV_2D:
                case GGML_OP_CONV_3D:
                    {
                        cur = GGML_IM2COL_WORK_SIZE;
                    } break;
                case GGML_OP_CONV_TRANSPOSE_2D:
                    {
                        const int64_t ne00 = node->src[0]->ne[0]; // W
                        const int64_t ne01 = node->src[0]->ne[1]; // H
                        const int64_t ne02 = node->src[0]->ne[2]; // Channels Out
                        const int64_t ne03 = node->src[0]->ne[3]; // Channels In

                        const int64_t ne10 = node->src[1]->ne[0]; // W
                        const int64_t ne11 = node->src[1]->ne[1]; // H
                        const int64_t ne12 = node->src[1]->ne[2]; // Channels In

                        GGML_ASSERT(node->src[0]->type == GGML_TYPE_F16 || node->src[0]->type == GGML_TYPE_F32);
                        GGML_ASSERT(node->src[1]->type == GGML_TYPE_F32);

                        cur += ggml_type_size(node->src[0]->type) * ne00 * ne01 * ne02 * ne03;
                        cur += ggml_type_size(node->src[0]->type) * ne10 * ne11 * ne12;

                    } break;
                case GGML_OP_TOP_K:
                    {
                        cur += sizeof(int32_t)*node->src[0]->ne[0]*n_tasks;
                    } break;
                case GGML_OP_FLASH_ATTN_EXT:
                    {
                        const int64_t neq2 = node->src[0]->ne[2]; // number of query heads
                        const int64_t DK = node->src[1]->ne[0];
                        const int64_t DV = node->src[2]->ne[0];

                        // Tiled flash attention scratch (tile sizes defined in common.h)
                        // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                        size_t prefill  = sizeof(float)*(GGML_FA_TILE_Q*DK + 2*GGML_FA_TILE_Q*GGML_FA_TILE_KV + GGML_FA_TILE_Q*DV + GGML_FA_TILE_KV*DV + GGML_FA_TILE_KV*DK)*n_tasks;

                        // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                        // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                        size_t n_chunks = n_tasks;
                        size_t decode   = sizeof(float)*(neq2*n_chunks*(2+DV) + n_tasks*(DK + 2*DV));

                        cur += MAX(prefill, decode);
                    } break;
                case GGML_OP_FLASH_ATTN_BACK:
                    {
                        const int64_t    D = node->src[0]->ne[0];
                        const int64_t ne11 = ggml_up(node->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
                        const int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
                        if (node->src[1]->type == GGML_TYPE_F32) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        } else if (node->src[1]->type == GGML_TYPE_F16) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        } else if (node->src[1]->type == GGML_TYPE_BF16) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        }
                    } break;

                case GGML_OP_CROSS_ENTROPY_LOSS:
                    {
                        cur = ggml_type_size(node->type)*(n_tasks + node->src[0]->ne[0]*n_tasks);
                    } break;
                case GGML_OP_GATED_DELTA_NET:
                    {
                        const int64_t S_v = node->src[2]->ne[0];
                        cur = S_v * sizeof(float) * n_tasks;
                    } break;
                case GGML_OP_COUNT:
                    {
                        GGML_ABORT("fatal error");
                    }
                default:
                    break;
            }
        }

        work_size = MAX(work_size, cur);
    }

    if (work_size > 0) {
        work_size += CACHE_LINE_SIZE*(n_threads);
    }

    cplan.threadpool = threadpool;
    cplan.n_threads  = MIN(max_tasks, n_threads);
    cplan.work_size  = work_size;
    cplan.work_data  = NULL;

    return cplan;
}

static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool    * tp    = state->threadpool;

    const struct ggml_cgraph * cgraph = tp->cgraph;
    const struct ggml_cplan  * cplan  = tp->cplan;

    set_numa_thread_affinity(state->ith);

    struct ggml_compute_params params = {
        /*.ith        =*/ state->ith,
        /*.nth        =*/ atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK,
        /*.wsize      =*/ cplan->work_size,
        /*.wdata      =*/ cplan->work_data,
        /*.threadpool =*/ tp,
        /*.use_ref    =*/ cplan->use_ref,
    };
    const bool layer_gap_trace =
        state->ith == 0 && ggml_cpu_layer_gap_trace_enabled();
    uint64_t layer_gap_last_start_us = 0;
    struct ggml_tensor * layer_gap_last_node = NULL;

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p\n", state->ith, (const void *)cplan);
#else
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p last-graph %d\n", state->ith, (const void *)cplan, state->last_graph);
#endif

    for (int node_n = 0; node_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort, memory_order_relaxed) != node_n; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];

        if (ggml_op_is_empty(node->op)) {
            // skip NOPs
            continue;
        }

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        const uint64_t layer_gap_start_us =
            layer_gap_trace ? (uint64_t) ggml_time_us() : 0;
        const bool kimi_pipeline_checkpoint =
            ggml_kimi_pipeline_m1_checkpoint_wants(node);
        ggml_compute_forward(&params, node);

        if (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_data)) {
            atomic_store_explicit(&tp->abort, node_n + 1, memory_order_relaxed);
            tp->ec    = GGML_STATUS_ABORTED;
        }

        if (node_n + 1 < cgraph->n_nodes || kimi_pipeline_checkpoint) {
            ggml_barrier(state->threadpool);
        }
        if (kimi_pipeline_checkpoint) {
            if (state->ith == 0) {
                ggml_kimi_pipeline_m1_checkpoint_record(node);
            }
            // The next CPU node may reuse this scratch immediately.  Keep all
            // worker threads behind the byte-exact checkpoint until hashing is
            // complete; this barrier exists only in the M1 spike process.
            ggml_barrier(state->threadpool);
        }
        if (layer_gap_trace) {
            if (node_n + 1 < cgraph->n_nodes) {
                ggml_cpu_layer_gap_trace_node(
                    layer_gap_start_us,
                    (uint64_t) ggml_time_us(),
                    (int) node->op,
                    ggml_op_name(node->op),
                    node->name);
            } else {
                layer_gap_last_start_us = layer_gap_start_us;
                layer_gap_last_node = node;
            }
        }
    }

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p\n", state->ith, (const void *)cplan);
#else
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p last-graph %d\n", state->ith, (const void *)cplan, state->last_graph);
#endif

    ggml_barrier(state->threadpool);
    if (layer_gap_trace && layer_gap_last_node != NULL) {
        ggml_cpu_layer_gap_trace_node(
            layer_gap_last_start_us,
            (uint64_t) ggml_time_us(),
            (int) layer_gap_last_node->op,
            ggml_op_name(layer_gap_last_node->op),
            layer_gap_last_node->name);
    }

    return 0;
}

#ifndef GGML_USE_OPENMP

// check if thread is ready to proceed (exit from polling or sleeping)
// returns true if loops should exit, sets state->pending to indicate new work
static inline bool ggml_graph_compute_thread_ready(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    if (state->pending || threadpool->stop || threadpool->pause) { return true; }

    // check for new graph/work
    int n_graph   = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed);
    int n_threads = n_graph & GGML_THREADPOOL_N_THREADS_MASK;
    if (n_graph != state->last_graph) {
        state->pending    = (state->ith < n_threads);
        state->last_graph = n_graph;
        return true;
    }

    return false;
}

// sync thread state after polling
static inline void ggml_graph_compute_thread_sync(struct ggml_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&state->threadpool->n_graph, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
    UNUSED(state);
}

static inline bool ggml_graph_compute_poll_for_work(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    const uint64_t n_rounds = 1024UL * 128 * threadpool->poll;

    for (uint64_t i=0; !ggml_graph_compute_thread_ready(state) && i < n_rounds; i++) {
        // No new work. Keep polling.
        ggml_thread_cpu_relax();
    }

    return state->pending;
}

static inline bool ggml_graph_compute_check_for_work(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    if (ggml_graph_compute_poll_for_work(state)) {
        ggml_graph_compute_thread_sync(state);
        return state->pending;
    }

    ggml_mutex_lock_shared(&threadpool->mutex);
    while (!ggml_graph_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    ggml_mutex_unlock_shared(&threadpool->mutex);

    return state->pending;
}

static thread_ret_t ggml_graph_compute_secondary_thread(void* data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool * threadpool = state->threadpool;

    ggml_thread_apply_priority(threadpool->prio);
    if (ggml_thread_cpumask_is_valid(state->cpumask)) {
        ggml_thread_apply_affinity(state->cpumask);
    }

    while (true) {
        // Check if we need to sleep
        while (threadpool->pause) {
            GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            ggml_mutex_lock_shared(&threadpool->mutex);
            if (threadpool->pause) {
                ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        if (threadpool->stop) break;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        ggml_graph_compute_check_for_work(state);
        if (state->pending) {
            state->pending = false;
            ggml_graph_compute_thread(state);
        }
    }

    return (thread_ret_t) 0;
}

// Start processing new graph
static void ggml_graph_compute_kickoff(struct ggml_threadpool * threadpool, int n_threads)
{
    // Always take the mutex here because the worker threads are doing hybrid poll/wait

    ggml_mutex_lock(&threadpool->mutex);

    // Update the number of active threads and the graph count
    int n_graph = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed) >> GGML_THREADPOOL_N_THREADS_BITS;
    n_graph = ((n_graph + 1) << GGML_THREADPOOL_N_THREADS_BITS) | (n_threads & GGML_THREADPOOL_N_THREADS_MASK);

    GGML_PRINT_DEBUG("compute-kickoff: n_threads %d n_graph %d\n", n_threads, n_graph);

    // Indicate the graph is ready to be processed
    // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
    atomic_store_explicit(&threadpool->n_graph, n_graph, memory_order_seq_cst);

    if (threadpool->pause) {
       // Update main thread prio and affinity to match the threadpool settings
       ggml_thread_apply_priority(threadpool->prio);
       if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       ggml_threadpool_resume_locked(threadpool);
    } else {
       ggml_cond_broadcast(&threadpool->cond);
    }

    ggml_mutex_unlock(&threadpool->mutex);
}

#endif // GGML_USE_OPENMP

static struct ggml_threadpool * ggml_threadpool_new_impl(
    struct ggml_threadpool_params * tpp,
               struct ggml_cgraph * cgraph,
                struct ggml_cplan * cplan) {

    struct ggml_threadpool * threadpool =
        ggml_aligned_malloc(sizeof(struct ggml_threadpool));
    {
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->n_graph          = 0;
        threadpool->n_barrier        = 0;
        threadpool->n_barrier_passed = 0;
        threadpool->current_chunk    = 0;
        threadpool->stop             = false;
        threadpool->pause            = tpp->paused;
        threadpool->abort            = -1;
        threadpool->workers          = NULL;
        threadpool->n_threads        = tpp->n_threads;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

    // Allocate and init workers state
    const size_t workers_size = sizeof(struct ggml_compute_state) * tpp->n_threads;
    struct ggml_compute_state * workers = ggml_aligned_malloc(workers_size);

    memset(workers, 0, workers_size);
    for (int j = 0; j < tpp->n_threads; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

#ifdef GGML_USE_OPENMP
    int32_t cpumask_iter = 0;

    // Compute CPU masks for each thread
    for (int j = 0; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);
    }
#else // GGML_USE_OPENMP
    ggml_mutex_init(&threadpool->mutex);
    ggml_cond_init(&threadpool->cond);

    // Spin the threads for all workers, and update CPU placements.
    // Place the main thread last (towards the higher numbered CPU cores).

    int32_t cpumask_iter = 0;

    for (int j = 1; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);

        int32_t rc = ggml_thread_create(&workers[j].thrd, NULL, ggml_graph_compute_secondary_thread, &workers[j]);
        GGML_ASSERT(rc == 0);
    }

    ggml_thread_cpumask_next(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);

    if (!threadpool->pause) {
        // Update main thread prio and affinity at the start, otherwise we'll do it in resume
        ggml_thread_apply_priority(threadpool->prio);
        if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
    }
#endif // GGML_USE_OPENMP

    return threadpool;
}

struct ggml_threadpool * ggml_threadpool_new(struct ggml_threadpool_params * tpp) {
    return ggml_threadpool_new_impl(tpp, NULL, NULL);
}

enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    ggml_cpu_init();

    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_threads > 0);
    GGML_ASSERT(cplan->work_size == 0 || cplan->work_data != NULL);

    int n_threads                               = cplan->n_threads;
    struct ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    if (threadpool == NULL) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        struct ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);
        threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
    } else {
        // Reset some of the parameters that need resetting
        // No worker threads should be accessing the parameters below at this stage
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->abort            = -1;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

#ifdef GGML_USE_OPENMP
    if (n_threads > 1) {
        #pragma omp parallel num_threads(n_threads)
        {
            #pragma omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_threads = omp_get_num_threads();
                atomic_store_explicit(&threadpool->n_graph, n_threads, memory_order_relaxed);
            }

            // Apply thread CPU mask and priority
            int ith = omp_get_thread_num();

            ggml_thread_apply_priority(threadpool->prio);
            if (ggml_thread_cpumask_is_valid(threadpool->workers[ith].cpumask)) {
                ggml_thread_apply_affinity(threadpool->workers[ith].cpumask);
            }
            ggml_graph_compute_thread(&threadpool->workers[ith]);
        }
    } else {
        atomic_store_explicit(&threadpool->n_graph, 1, memory_order_relaxed);
        ggml_graph_compute_thread(&threadpool->workers[0]);
    }
#else
    if (n_threads > threadpool->n_threads) {
        GGML_LOG_WARN("cplan requested more threads (%d) than available (%d)\n", n_threads, threadpool->n_threads);
        n_threads = threadpool->n_threads;
    }

    // Kick all threads to start the new graph
    ggml_graph_compute_kickoff(threadpool, n_threads);

    // This is a work thread too
    ggml_graph_compute_thread(&threadpool->workers[0]);
#endif

    // don't leave affinity set on the main thread
    clear_numa_thread_affinity();

    enum ggml_status ret = threadpool->ec;

    if (disposable_threadpool) {
        ggml_threadpool_free(threadpool);
    }

    return ret;
}

enum ggml_status ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads) {
    struct ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads, NULL);

    cplan.work_data = (uint8_t *)ggml_new_buffer(ctx, cplan.work_size);

    return ggml_graph_compute(cgraph, &cplan);
}

void ggml_cpu_fp32_to_fp32(const float * x, float * y, int64_t n) {
    memcpy(y, x, n * sizeof(float));
}

void ggml_cpu_fp32_to_fp16(const float * x, ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m256i y_vec = _mm512_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), y_vec);
    }
#endif
    for (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    for (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#elif defined(__riscv_zvfh)
    for (int vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
        __riscv_vse16_v_f16m1((_Float16 *)&y[i], vy, vl);
    }
#endif
    for (; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(x[i]);
    }
}

void ggml_cpu_fp16_to_fp32(const ggml_fp16_t * x, float * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m256i x_vec = _mm256_loadu_si256((const __m256i *)(x + i));
        __m512 y_vec = _mm512_cvtph_ps(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
#endif
    for (; i + 7 < n; i += 8) {
        __m128i x_vec = _mm_loadu_si128((const __m128i *)(x + i));
        __m256 y_vec = _mm256_cvtph_ps(x_vec);
        _mm256_storeu_ps(y + i, y_vec);
    }
    for (; i + 3 < n; i += 4) {
        __m128i x_vec = _mm_loadl_epi64((const __m128i *)(x + i));
        __m128 y_vec = _mm_cvtph_ps(x_vec);
        _mm_storeu_ps(y + i, y_vec);
    }

#elif defined(__riscv_v_intrinsic) && defined(__riscv_zvfhmin)
    // calculate step size
    const int epr = __riscv_vsetvlmax_e16m2();
    const int step = epr * 2;
    const int np = (n & ~(step - 1));

    // unroll by 2
    for (; i < np; i += step) {
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((const _Float16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vfloat16m2_t ax1 = __riscv_vle16_v_f16m2((const _Float16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvt_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    int vl;
    for (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((const _Float16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }

#endif

    for (; i < n; ++i) {
        y[i] = GGML_CPU_FP16_TO_FP32(x[i]);
    }
}

void ggml_cpu_fp32_to_bf16(const float * x, ggml_bf16_t * y, int64_t n) {
    int64_t i = 0;
    for (; i < n; ++i) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

void ggml_cpu_fp32_to_i32(const float * x, int32_t * y, int64_t n) {
    int64_t i = 0;
    for (; i < n; ++i) {
        y[i] = x[i];
    }
}

void ggml_cpu_bf16_to_fp32(const ggml_bf16_t * x, float * y, int64_t n) {
    int64_t i = 0;
#if defined(__AVX2__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i,
                        _mm512_castsi512_ps(
                            _mm512_slli_epi32(
                                _mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(
                                        (const __m256i *)(x + i))),
                                16)));
    }
#endif
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i,
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(
                                    _mm_loadu_si128(
                                        (const __m128i *)(x + i))),
                                16)));
    }
#elif defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfmin)
    // calculate step size
    const int epr = __riscv_vsetvlmax_e16m2();
    const int step = epr * 2;
    const int np = (n & ~(step - 1));

    // unroll by 2
    for (; i < np; i += step) {
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((const __bf16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vbfloat16m2_t ax1 = __riscv_vle16_v_bf16m2((const __bf16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    int vl;
    for (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((const __bf16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }
#endif
    for (; i < n; i++) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

int ggml_cpu_has_avx(void) {
#if defined(__AVX__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx_vnni(void) {
#if defined(__AVXVNNI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx2(void) {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512(void) {
#if defined(__AVX512F__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_vbmi(void) {
#if defined(__AVX512VBMI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_vnni(void) {
#if defined(__AVX512VNNI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_bf16(void) {
#if defined(__AVX512BF16__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_amx_int8(void) {
#if defined(__AMX_INT8__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_bmi2(void) {
#if defined(__BMI2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_fma(void) {
#if defined(__FMA__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_arm_fma(void) {
#if defined(__ARM_FEATURE_FMA)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_riscv_v(void) {
#if defined(__riscv_v_intrinsic)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_get_rvv_vlen(void) {
#if defined(__riscv) && defined(__riscv_v_intrinsic)
    return ggml_riscv_arch_features.rvv_vlen;
#else
    return 0;
#endif
}

int ggml_cpu_has_f16c(void) {
#if defined(__F16C__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_fp16_va(void) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_wasm_simd(void) {
#if defined(__wasm_simd128__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_llamafile(void) {
#if defined(GGML_USE_LLAMAFILE)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sse3(void) {
#if defined(__SSE3__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_ssse3(void) {
#if defined(__SSSE3__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_vsx(void) {
#if defined(__POWER9_VECTOR__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_vxe(void) {
#if defined(__VXE__) || defined(__VXE2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_neon(void) {
#if defined(__ARM_ARCH) && defined(__ARM_NEON)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_dotprod(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sve(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_matmul_int8(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_MATMUL_INT8)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_get_sve_cnt(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    return ggml_arm_arch_features.sve_cnt;
#else
    return 0;
#endif
}

int ggml_cpu_has_sme(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME)
    return 1;
#else
    return 0;
#endif
}

void ggml_cpu_init(void) {
    // needed to initialize ggml_time
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    ggml_critical_section_start();

    static bool is_first_call = true;

    if (is_first_call) {
        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

            for (int i = 0; i < (1 << 16); ++i) {
                union {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                float f = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
                ggml_table_f32_f16[i] = f;
                ggml_table_gelu_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_f32(f));
                ggml_table_gelu_quick_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_quick_f32(f));
            }

            // initialize E8M0 half table (256 entries)
            for (int i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_e8m0_half[i] = GGML_E8M0_TO_FP32_HALF(i);
            }

            const uint64_t t_end = ggml_time_us(); UNUSED(t_end);

            GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0);

#ifdef GGML_USE_OPENMP
            //if (!getenv("OMP_WAIT_POLICY")) {
            //    // set the wait policy to active, so that OpenMP threads don't sleep
            //    setenv("OMP_WAIT_POLICY", "active", 0)
            //}

            if (!getenv("KMP_BLOCKTIME")) {
                // set the time to wait before sleeping a thread
                // this is less aggressive than setting the wait policy to active, but should achieve similar results in most cases
#ifdef _WIN32
                _putenv_s("KMP_BLOCKTIME", "200"); // 200ms
#else
                setenv("KMP_BLOCKTIME", "200", 0); // 200ms
#endif
            }
#endif
        }

#if defined(__ARM_ARCH)
        ggml_init_arm_arch_features();
#endif

#if defined(__riscv)
        ggml_init_riscv_arch_features();
#endif

        is_first_call = false;
    }

    ggml_critical_section_end();
}
