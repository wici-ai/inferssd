// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"
#include "ggml-moe-pack-plan.h"

#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <condition_variable>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef __APPLE__
#include <dlfcn.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#if !defined(__APPLE__) && (defined(__GNUC__) || defined(__clang__))
extern "C" bool ggml_cuda_moe_expert_pack_read_to_host(
        const char * tensor_name,
        int expert_idx,
        size_t nbytes,
        void * dst,
        size_t dst_capacity,
        size_t * nread,
        int * source_is_gguf) __attribute__((weak));
extern "C" bool ggml_cuda_moe_expert_pack_copy_ranges_to_tensor(
        ggml_backend_t backend,
        ggml_tensor * tensor,
        const char * tensor_name,
        const int32_t * range_first,
        const int32_t * range_last,
        size_t range_count,
        int64_t n_expert,
        size_t expert_bytes,
        size_t padding,
        bool cache_bypass,
        uint64_t * experts_copied,
        uint64_t * iouring_reads,
        uint64_t * iouring_bytes,
        uint64_t * cache_lookups,
        uint64_t * cache_hits) __attribute__((weak));
extern "C" bool ggml_cuda_moe_dense_has_tensor(
        const char * tensor_name,
        size_t nbytes,
        size_t * block_bytes) __attribute__((weak));
extern "C" bool ggml_cuda_moe_dense_read_ahead(
        const char * tensor_name,
        size_t nbytes) __attribute__((weak));
extern "C" bool ggml_cuda_moe_dense_copy_to_tensor(
        ggml_backend_t backend,
        ggml_tensor * tensor,
        const char * tensor_name,
        bool * read_ahead_hit) __attribute__((weak));
extern "C" void ggml_cuda_moe_seed_offer(
        ggml_backend_t backend,
        const ggml_tensor * tensor,
        const char * tensor_name,
        int64_t n_expert,
        size_t expert_bytes,
        const int32_t * counts) __attribute__((weak));
extern "C" void ggml_cuda_moe_seed_fence(ggml_backend_t backend) __attribute__((weak));
#endif

using ggml_cuda_moe_expert_pack_read_to_host_fn = bool (*)(
        const char * tensor_name,
        int expert_idx,
        size_t nbytes,
        void * dst,
        size_t dst_capacity,
        size_t * nread,
        int * source_is_gguf);

static ggml_cuda_moe_expert_pack_read_to_host_fn ggml_cuda_moe_expert_pack_read_to_host_resolve() {
#if defined(__APPLE__)
    return reinterpret_cast<ggml_cuda_moe_expert_pack_read_to_host_fn>(
            dlsym(RTLD_DEFAULT, "ggml_cuda_moe_expert_pack_read_to_host"));
#elif defined(__GNUC__) || defined(__clang__)
    return ggml_cuda_moe_expert_pack_read_to_host;
#else
    return nullptr;
#endif
}

using ggml_cuda_moe_expert_pack_copy_ranges_to_tensor_fn = bool (*)(
        ggml_backend_t,
        ggml_tensor *,
        const char *,
        const int32_t *,
        const int32_t *,
        size_t,
        int64_t,
        size_t,
        size_t,
        bool,
        uint64_t *,
        uint64_t *,
        uint64_t *,
        uint64_t *,
        uint64_t *);

static ggml_cuda_moe_expert_pack_copy_ranges_to_tensor_fn
ggml_cuda_moe_expert_pack_copy_ranges_to_tensor_resolve() {
#if defined(__APPLE__)
    return reinterpret_cast<ggml_cuda_moe_expert_pack_copy_ranges_to_tensor_fn>(
            dlsym(RTLD_DEFAULT, "ggml_cuda_moe_expert_pack_copy_ranges_to_tensor"));
#elif defined(__GNUC__) || defined(__clang__)
    return ggml_cuda_moe_expert_pack_copy_ranges_to_tensor;
#else
    return nullptr;
#endif
}

// the expert transport's index of the dense weights homed on the drive (tools/adaptive DENSE_HOMES): whether it holds
// a weight, the read of the next one ahead, the copy of one to its device tensor
using ggml_cuda_moe_dense_has_tensor_fn = bool (*)(const char *, size_t, size_t *);
using ggml_cuda_moe_dense_read_ahead_fn = bool (*)(const char *, size_t);
using ggml_cuda_moe_dense_copy_to_tensor_fn = bool (*)(ggml_backend_t, ggml_tensor *, const char *, bool *);

#if defined(__APPLE__)
#define GGML_KIMI_RESOLVE(name) reinterpret_cast<name##_fn>(dlsym(RTLD_DEFAULT, #name))
#elif defined(__GNUC__) || defined(__clang__)
#define GGML_KIMI_RESOLVE(name) static_cast<name##_fn>(name)
#else
#define GGML_KIMI_RESOLVE(name) static_cast<name##_fn>(nullptr)
#endif

static ggml_cuda_moe_dense_has_tensor_fn ggml_cuda_moe_dense_has_tensor_resolve() {
    return GGML_KIMI_RESOLVE(ggml_cuda_moe_dense_has_tensor);
}

static ggml_cuda_moe_dense_read_ahead_fn ggml_cuda_moe_dense_read_ahead_resolve() {
    return GGML_KIMI_RESOLVE(ggml_cuda_moe_dense_read_ahead);
}

static ggml_cuda_moe_dense_copy_to_tensor_fn ggml_cuda_moe_dense_copy_to_tensor_resolve() {
    return GGML_KIMI_RESOLVE(ggml_cuda_moe_dense_copy_to_tensor);
}

// Prefill seeding of the expert cache (ggml-cuda.h): the scheduler offers each routed expert tensor it copied for a
// graph pass, and fences its backend's stream behind the cache's copies before anything can reuse the source.
using ggml_cuda_moe_seed_offer_fn = void (*)(ggml_backend_t, const ggml_tensor *, const char *, int64_t, size_t,
        const int32_t *);
using ggml_cuda_moe_seed_fence_fn = void (*)(ggml_backend_t);

static ggml_cuda_moe_seed_offer_fn ggml_cuda_moe_seed_offer_resolve() {
    return GGML_KIMI_RESOLVE(ggml_cuda_moe_seed_offer);
}

static ggml_cuda_moe_seed_fence_fn ggml_cuda_moe_seed_fence_resolve() {
    return GGML_KIMI_RESOLVE(ggml_cuda_moe_seed_fence);
}

// backend buffer type

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    return buft->iface.alloc_buffer(buft, size);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        assert(size >= ggml_nbytes(tensor));
        return size;
    }
    return ggml_nbytes(tensor);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i      iface,
               void *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = new ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}

const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    if (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        return NULL;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    if (!buffer->iface.get_base) {
        return NULL;
    }

    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    }
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->buft;
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    if (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->guid;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    if (backend->iface.set_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    if (backend->iface.get_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_set_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_set_2d(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.set_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.set_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != NULL);

    return backend->iface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != NULL);

    backend->iface.graph_plan_free(backend, plan);
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != NULL);

    return backend->iface.graph_plan_compute(backend, plan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_op(backend->device, op);
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_buft(backend->device, buft);
}

bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return backend->device;
}

// backend copy

void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    GGML_ASSERT(backend_dst);
    if (backend_dst->iface.cpy_tensor_async != NULL) {
        if (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            return;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    if (device == NULL || device->iface.event_new == NULL) {
        return NULL;
    }
    return device->iface.event_new(device);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event == NULL) {
        return;
    }
    event->device->iface.event_free(event->device, event);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != NULL);

    backend->iface.event_record(backend, event);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != NULL);

    backend->iface.event_wait(backend, event);
}

static void ggml_backend_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    if (backend->iface.graph_optimize != NULL) {
        backend->iface.graph_optimize(backend, cgraph);
    }
}

// Backend device

const char * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_name(device);
}

const char * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_description(device);
}

void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, free, total);
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_type(device);
}

void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, sizeof(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    GGML_ASSERT(device);
    return device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    if (device->iface.get_host_buffer_type == NULL) {
        return NULL;
    }

    return device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    return device->iface.supports_op(device, op);
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    return device->iface.supports_buft(device, buft);
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    if (device->iface.offload_op != NULL) {
        return device->iface.offload_op(device, op);
    }

    return false;
}

// Backend (reg)

const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_device_count(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    return reg->iface.get_device(reg, index);
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_ASSERT(reg);
    if (!reg->iface.get_proc_address) {
        return NULL;
    }
    return reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

struct ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

static void ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    free(ctx->buffers);
    free(ctx);
}

static void ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

static const struct ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ NULL,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ NULL,
    /* .get_tensor      = */ NULL,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ NULL,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(sizeof(struct ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * sizeof(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != NULL);

    size_t total_size = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    return ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor * inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_inputs;
    // graph view of this split
    struct ggml_cgraph graph;
};

enum ggml_backend_sched_moe_prefetch_policy {
    GGML_BACKEND_SCHED_MOE_PREFETCH_NONE,
    GGML_BACKEND_SCHED_MOE_PREFETCH_SETMARKOV,
};

struct ggml_backend_sched_moe_transition {
    std::vector<int32_t> from;
    std::vector<uint32_t> counts;
};

struct ggml_backend_sched_moe_cache {
    const ggml_tensor * input;
    int backend_id;
    int n_expert;
    int n_slots;

    size_t expert_size;
    size_t slot_padding;
    size_t slot_stride;
    size_t weights_size;
    size_t ids_nbytes;

    ggml_backend_buffer_t weights_buffer;
    ggml_tensor weights_tensor;

    ggml_backend_buffer_t ids_buffer;
    ggml_tensor ids_tensor;

    std::vector<int32_t> slot_of;
    std::vector<int32_t> expert_in_slot;
    std::vector<uint8_t> slot_speculative;
    std::vector<uint64_t> lru_tick;
    std::vector<int32_t> remapped_ids;
    std::vector<int32_t> previous_experts;
    std::vector<ggml_backend_sched_moe_transition> transitions;
    uint64_t now;
    uint64_t hits;
    uint64_t misses;
    uint64_t bypasses;
    uint64_t bytes_copied;
    uint64_t speculative_hits;
    uint64_t prefetches;
    uint64_t prefetch_evictions;
    uint64_t wrong_prefetches;
    uint64_t bytes_prefetched;
};

struct ggml_backend_sched_moe_restore {
    ggml_tensor * node;
    ggml_tensor * src0;
    ggml_tensor * src2;
};

enum ggml_backend_sched_kimi_async_d2h_edge_state {
    GGML_KIMI_ASYNC_D2H_PLANNED = 0,
    GGML_KIMI_ASYNC_D2H_RESERVED,
    GGML_KIMI_ASYNC_D2H_SOURCE_PUBLISHED,
    GGML_KIMI_ASYNC_D2H_TRANSFER_ENQUEUED,
    GGML_KIMI_ASYNC_D2H_READY_PUBLISHED,
    GGML_KIMI_ASYNC_D2H_HOST_COPIED,
    GGML_KIMI_ASYNC_D2H_CONSUMER_QUEUED,
    GGML_KIMI_ASYNC_D2H_RETIRED,
    GGML_KIMI_ASYNC_D2H_POISONED,
};

struct ggml_backend_sched_kimi_async_d2h_edge {
    ggml_tensor * src = nullptr;
    ggml_tensor * dst = nullptr;
    ggml_backend_t src_backend = nullptr;
    ggml_backend_buffer_t src_buffer = nullptr;
    ggml_backend_buffer_t dst_buffer = nullptr;
    int producer_split = -1;
    int consumer_split = -1;
    int input_id = -1;
    int slot = -1;
    size_t bytes = 0;
    uint64_t generation = 0;
    enum ggml_backend_sched_kimi_async_d2h_edge_state state =
        GGML_KIMI_ASYNC_D2H_PLANNED;
};

struct ggml_backend_sched_kimi_async_d2h_task {
    size_t edge = 0;
    int slot = -1;
    uint64_t generation = 0;
    ggml_tensor * src = nullptr;
    size_t bytes = 0;
};

struct ggml_backend_sched_kimi_async_d2h_slot {
    ggml_backend_buffer_t staging = nullptr;
    ggml_backend_event_t producer_ready = nullptr;
    ggml_backend_event_t d2h_done = nullptr;
    uint64_t generation = 0;
    bool busy = false;
};

struct ggml_backend_sched_kimi_async_d2h_state {
    static constexpr int N_SLOTS = 2;
    static constexpr int EXPECTED_SPLITS = 252;
    static constexpr int EXPECTED_EDGES = 219;

    std::mutex mutex;
    std::condition_variable work_cv;
    std::condition_variable state_cv;
    std::deque<ggml_backend_sched_kimi_async_d2h_task> tasks;
    std::thread worker;
    bool worker_started = false;
    bool stop = false;
    bool failed = false;
    bool active_graph = false;
    bool decline_logged = false;
    const char * failure_reason = "none";
    const char * last_decline_reason = "none";
    size_t active_tasks = 0;

    ggml_backend_dev_t device = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t host_buft = nullptr;
    size_t staging_capacity = 0;
    ggml_backend_sched_kimi_async_d2h_slot slots[N_SLOTS];

    std::vector<ggml_backend_sched_kimi_async_d2h_edge> edges;
    std::vector<int> edge_by_split_input;
    size_t next_submit = 0;
    int producer_frontier = -1;
    uint64_t graph_epoch = 0;
    uint64_t next_generation = 0;

    uint64_t graphs_started = 0;
    uint64_t graphs_activated = 0;
    uint64_t graphs_declined = 0;
    uint64_t graphs_completed = 0;
    uint64_t graphs_poisoned = 0;
    uint64_t descriptors_planned = 0;
    uint64_t descriptors_reserved = 0;
    uint64_t descriptors_submitted = 0;
    uint64_t transfer_enqueued = 0;
    uint64_t ready_published = 0;
    uint64_t consumer_queued = 0;
    uint64_t descriptors_retired = 0;
    uint64_t producer_records = 0;
    uint64_t producer_waits = 0;
    uint64_t done_records = 0;
    uint64_t done_host_waits = 0;
    uint64_t d2h_calls = 0;
    uint64_t d2h_bytes = 0;
    uint64_t host_publish_calls = 0;
    uint64_t host_publish_bytes = 0;
    uint64_t scheduler_ready_waits = 0;
    uint64_t scheduler_ready_wait_us = 0;
    uint64_t scheduler_ready_wait_max_us = 0;
    uint64_t deferred_submissions = 0;
    uint64_t worker_wakeups = 0;
    uint64_t queue_max = 0;
    uint64_t stale_generation_failures = 0;
    uint64_t identity_failures = 0;
    uint64_t layout_failures = 0;
    uint64_t ownership_failures = 0;
    uint64_t reorder_failures = 0;
    uint64_t init_failures = 0;
};

enum ggml_backend_sched_kimi_async_h2d_edge_state {
    GGML_KIMI_ASYNC_H2D_PLANNED = 0,
    GGML_KIMI_ASYNC_H2D_RESERVED,
    GGML_KIMI_ASYNC_H2D_SOURCE_SNAPSHOTTED,
    GGML_KIMI_ASYNC_H2D_RELEASE_PUBLISHED,
    GGML_KIMI_ASYNC_H2D_TRANSFER_ENQUEUED,
    GGML_KIMI_ASYNC_H2D_READY_PUBLISHED,
    GGML_KIMI_ASYNC_H2D_CONSUMER_QUEUED,
    GGML_KIMI_ASYNC_H2D_RETIRED,
    GGML_KIMI_ASYNC_H2D_POISONED,
};

struct ggml_backend_sched_kimi_async_h2d_edge {
    ggml_tensor * src = nullptr;
    ggml_tensor * dst = nullptr;
    ggml_backend_t src_backend = nullptr;
    ggml_backend_t dst_backend = nullptr;
    ggml_backend_buffer_t src_buffer = nullptr;
    ggml_backend_buffer_t dst_buffer = nullptr;
    int producer_split = -1;
    int consumer_split = -1;
    int input_id = -1;
    int slot = -1;
    size_t bytes = 0;
    bool conservative_source = false;
    uint64_t generation = 0;
    enum ggml_backend_sched_kimi_async_h2d_edge_state state =
        GGML_KIMI_ASYNC_H2D_PLANNED;
};

struct ggml_backend_sched_kimi_async_h2d_task {
    size_t edge = 0;
    int slot = -1;
    uint64_t generation = 0;
    ggml_tensor * src = nullptr;
    ggml_tensor * dst = nullptr;
    ggml_backend_t src_backend = nullptr;
    size_t bytes = 0;
    bool reuse_slot = false;
    bool transfer_phase = false;
};

struct ggml_backend_sched_kimi_async_h2d_slot {
    ggml_backend_buffer_t staging = nullptr;
    ggml_backend_event_t destination_release = nullptr;
    ggml_backend_event_t input_ready = nullptr;
    uint64_t generation = 0;
    bool busy = false;
    bool has_previous = false;
};

struct ggml_backend_sched_kimi_async_h2d_state {
    static constexpr int N_SLOTS = 2;
    static constexpr int EXPECTED_SPLITS = 252;
    static constexpr int EXPECTED_EDGES = 206;

    std::mutex mutex;
    std::condition_variable work_cv;
    std::condition_variable state_cv;
    std::deque<ggml_backend_sched_kimi_async_h2d_task> tasks;
    std::thread worker;
    bool worker_started = false;
    bool stop = false;
    bool failed = false;
    bool active_graph = false;
    bool decline_logged = false;
    const char * failure_reason = "none";
    const char * last_decline_reason = "none";
    size_t active_tasks = 0;

    ggml_backend_dev_t device = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t host_buft = nullptr;
    size_t staging_capacity = 0;
    ggml_backend_sched_kimi_async_h2d_slot slots[N_SLOTS];

    std::vector<ggml_backend_sched_kimi_async_h2d_edge> edges;
    std::vector<int> edge_by_split_input;
    size_t next_submit = 0;
    int producer_frontier = -1;
    uint64_t graph_epoch = 0;
    uint64_t next_generation = 0;

    uint64_t graphs_started = 0;
    uint64_t graphs_activated = 0;
    uint64_t graphs_declined = 0;
    uint64_t graphs_completed = 0;
    uint64_t graphs_poisoned = 0;
    uint64_t descriptors_planned = 0;
    uint64_t descriptors_reserved = 0;
    uint64_t descriptors_submitted = 0;
    uint64_t mapped_sources = 0;
    uint64_t conservative_sources = 0;
    uint64_t source_syncs = 0;
    uint64_t snapshot_calls = 0;
    uint64_t snapshot_bytes = 0;
    uint64_t snapshot_waits = 0;
    uint64_t snapshot_wait_us = 0;
    uint64_t snapshot_wait_max_us = 0;
    uint64_t release_records = 0;
    uint64_t release_waits = 0;
    uint64_t transfer_enqueued = 0;
    uint64_t ready_records = 0;
    uint64_t ready_published = 0;
    uint64_t ready_waits = 0;
    uint64_t h2d_calls = 0;
    uint64_t h2d_bytes = 0;
    uint64_t consumer_queued = 0;
    uint64_t descriptors_retired = 0;
    uint64_t reuse_host_waits = 0;
    uint64_t worker_wakeups = 0;
    uint64_t queue_max = 0;
    uint64_t stale_generation_failures = 0;
    uint64_t identity_failures = 0;
    uint64_t layout_failures = 0;
    uint64_t ownership_failures = 0;
    uint64_t reorder_failures = 0;
    uint64_t init_failures = 0;
};

enum ggml_backend_sched_kimi_async_ids_plan_state {
    GGML_KIMI_ASYNC_IDS_PLANNED = 0,
    GGML_KIMI_ASYNC_IDS_RESERVED,
    GGML_KIMI_ASYNC_IDS_SOURCE_PUBLISHED,
    GGML_KIMI_ASYNC_IDS_TRANSFER_ENQUEUED,
    GGML_KIMI_ASYNC_IDS_HOST_READY,
    GGML_KIMI_ASYNC_IDS_CONSUMED,
    GGML_KIMI_ASYNC_IDS_RETIRED,
    GGML_KIMI_ASYNC_IDS_POISONED,
};

struct ggml_backend_sched_kimi_async_ids_plan {
    ggml_tensor * ids_tensor = nullptr;
    ggml_backend_t ids_backend = nullptr;
    ggml_backend_buffer_t ids_buffer = nullptr;
    int producer_split = -1;
    int consumer_split = -1;
    int consumer_input = -1;
    int slot = -1;
    int64_t n_expert = 0;
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    size_t nb0 = 0;
    size_t nb1 = 0;
    size_t bytes = 0;
    uint64_t generation = 0;
    enum ggml_backend_sched_kimi_async_ids_plan_state state =
        GGML_KIMI_ASYNC_IDS_PLANNED;
};

struct ggml_backend_sched_kimi_async_ids_task {
    size_t plan = 0;
    int slot = -1;
    uint64_t generation = 0;
};

struct ggml_backend_sched_kimi_async_ids_slot {
    ggml_backend_buffer_t staging = nullptr;
    ggml_backend_event_t producer_ready = nullptr;
    ggml_backend_event_t d2h_done = nullptr;
    std::vector<int32_t> ids;
    std::vector<int32_t> id_counts;
    std::vector<ggml_bitset_t> used_ids;
    uint64_t generation = 0;
    bool busy = false;
};

struct ggml_backend_sched_kimi_async_ids_state {
    static constexpr int N_SLOTS = 2;
    static constexpr int EXPECTED_SPLITS = 252;
    static constexpr int EXPECTED_PLANS = 90;
    static constexpr size_t EXPECTED_BYTES_PER_PLAN = 64;

    std::mutex mutex;
    std::condition_variable work_cv;
    std::condition_variable state_cv;
    std::deque<ggml_backend_sched_kimi_async_ids_task> tasks;
    std::thread worker;
    bool worker_started = false;
    bool stop = false;
    bool failed = false;
    bool active_graph = false;
    bool decline_logged = false;
    const char * failure_reason = "none";
    const char * last_decline_reason = "none";
    size_t active_tasks = 0;

    ggml_backend_dev_t device = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t host_buft = nullptr;
    size_t staging_capacity = 0;
    ggml_backend_sched_kimi_async_ids_slot slots[N_SLOTS];

    std::vector<ggml_backend_sched_kimi_async_ids_plan> plans;
    std::vector<int> plan_by_split_input;
    size_t next_submit = 0;
    int producer_frontier = -1;
    uint64_t graph_epoch = 0;
    uint64_t next_generation = 0;

    uint64_t graphs_started = 0;
    uint64_t graphs_activated = 0;
    uint64_t graphs_declined = 0;
    uint64_t graphs_completed = 0;
    uint64_t graphs_poisoned = 0;
    uint64_t plans_planned = 0;
    uint64_t plans_reserved = 0;
    uint64_t plans_submitted = 0;
    uint64_t producer_records = 0;
    uint64_t producer_waits = 0;
    uint64_t transfer_enqueued = 0;
    uint64_t d2h_calls = 0;
    uint64_t d2h_bytes = 0;
    uint64_t done_records = 0;
    uint64_t done_host_waits = 0;
    uint64_t plans_ready = 0;
    uint64_t parse_calls = 0;
    uint64_t parsed_elements = 0;
    uint64_t consumer_waits = 0;
    uint64_t consumer_wait_us = 0;
    uint64_t consumer_wait_max_us = 0;
    uint64_t plans_consumed = 0;
    uint64_t plans_retired = 0;
    uint64_t worker_wakeups = 0;
    uint64_t queue_max = 0;
    uint64_t slot_deferrals = 0;
    uint64_t stale_generation_failures = 0;
    uint64_t identity_failures = 0;
    uint64_t layout_failures = 0;
    uint64_t ownership_failures = 0;
    uint64_t reorder_failures = 0;
    uint64_t range_failures = 0;
    uint64_t init_failures = 0;
};

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    int next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    struct ggml_tensor * graph_inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_graph_inputs;

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;
    // the weights the expert transport's index holds as homed on the drive, by tensor (ggml_backend_sched_drive_home)
    std::unordered_map<const struct ggml_tensor *, bool> * drive_homes;
    std::vector<ggml_backend_sched_moe_cache *> * moe_caches;
    bool decode_phase;
    ggml_backend_sched_kimi_async_d2h_state * kimi_async_d2h;
    ggml_backend_sched_kimi_async_h2d_state * kimi_async_h2d;
    ggml_backend_sched_kimi_async_ids_state * kimi_async_ids;

    int debug;
    char (* debug_causes)[64];

    // used for debugging graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    int debug_realloc;
    int debug_graph_size;
    int debug_prev_graph_size;
};

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

static bool ggml_backend_sched_kimi_env_enabled(const char * name) {
    const char * env = getenv(name);
    return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0;
}


static const ggml_tensor * ggml_backend_sched_kimi_view_root(const ggml_tensor * tensor);
static const ggml_tensor * ggml_backend_sched_kimi_view_root(const ggml_tensor * tensor) {
    while (tensor != nullptr && tensor->view_src != nullptr) {
        tensor = tensor->view_src;
    }
    return tensor;
}

static constexpr uint8_t GGML_KIMI_RB_RELEASED = 1, GGML_KIMI_RB_PUBLISHED = 2, GGML_KIMI_RB_WAITED = 4;

// returns the priority of the backend, lower id is higher priority
static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}

#define SET_CAUSE(node, ...) do { \
    if (sched->debug_causes != nullptr) { \
        snprintf(sched->debug_causes[hash_id(node)], sizeof(sched->debug_causes[0]), __VA_ARGS__); \
    } \
} while (0)
#define GET_CAUSE(node) (sched->debug_causes != nullptr ? sched->debug_causes[hash_id(node)] : "")

static int ggml_backend_sched_backend_from_non_weight_src(
        ggml_backend_sched_t sched,
        ggml_tensor * tensor,
        int max_backend_id);

static ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(void);

// A weight homed on the drive (tools/adaptive DENSE_HOMES): a matrix in the CPU buffer that maps the GGUF which the
// expert transport's index holds in blocks.  The mapping is never read: the matrix product runs on the GPU at any
// batch size, and the weight's copy reads the blocks with O_DIRECT through the transport.  A weight in another host
// buffer (pinned RAM) keeps the scheduler's rule.  t is the weight or a view of all of it (a reshape); the weight
// when it is homed on the drive, else nullptr.
static const struct ggml_tensor * ggml_backend_sched_drive_home(ggml_backend_sched_t sched, const struct ggml_tensor * t) {
    const struct ggml_tensor * w = t->view_src ? t->view_src : t;
    if (w->buffer == NULL || ggml_n_dims(w) > 2 ||
            ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
            ggml_backend_buffer_get_type(w->buffer) != ggml_backend_cpu_buffer_from_ptr_type() ||
            (t != w && (t->view_offs != 0 || ggml_nbytes(t) != ggml_nbytes(w) || !ggml_is_contiguous(t)))) {
        return nullptr;
    }
    auto it = sched->drive_homes->find(w);
    if (it == sched->drive_homes->end()) {
        const auto has_tensor = ggml_cuda_moe_dense_has_tensor_resolve();
        it = sched->drive_homes->emplace(w, has_tensor && has_tensor(w->name, ggml_nbytes(w), nullptr)).first;
    }
    return it->second ? w : nullptr;
}

// the weight input copied after input input_id of split split_id, in the order the splits copy their inputs
static const struct ggml_tensor * ggml_backend_sched_next_weight_input(ggml_backend_sched_t sched, int split_id,
                                                                       int input_id) {
    for (int s = split_id; s < sched->n_splits; s++) {
        const struct ggml_backend_sched_split & split = sched->splits[s];
        for (int j = s == split_id ? input_id + 1 : 0; j < split.n_inputs; j++) {
            const struct ggml_tensor * in = split.inputs[j];
            const ggml_backend_buffer_t buf = in->view_src ? in->view_src->buffer : in->buffer;
            if (buf != NULL && ggml_backend_buffer_get_usage(buf) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(buf)) {
                return in;
            }
        }
    }
    return nullptr;
}

// returns the backend that should be used for the node based on the current locations
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const struct ggml_tensor * src = tensor->src[i];
        if (src == NULL) {
            continue;
        }
        // a weight homed on the drive (or a reshape of one): its matrix product runs on a GPU at any batch size, the
        // one its activations are on (the layer's) when that one supports it, else the first that does
        if (tensor->op == GGML_OP_MUL_MAT && i == 0 && ggml_backend_sched_drive_home(sched, src) != nullptr) {
            ggml_tensor * x = tensor->src[1];
            int b = x ? tensor_backend_id(x) : -1;
            if (b == -1 && x && x->view_src) {
                b = tensor_backend_id(x->view_src);
            }
            if (b < 0 || b >= sched->n_backends - 1 || !ggml_backend_supports_op(sched->backends[b], tensor)) {
                for (b = 0; b < sched->n_backends - 1; b++) {
                    if (ggml_backend_supports_op(sched->backends[b], tensor)) {
                        break;
                    }
                }
            }
            if (b < sched->n_backends - 1) {
                SET_CAUSE(tensor, "1.drv");
                return b;
            }
        }
        // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
        // not an ideal solution
        if (tensor->op != GGML_OP_ROPE && src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
            // check if a backend with higher prio wants to offload the op
            if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                const int non_weight_src_backend_id = ggml_backend_sched_backend_from_non_weight_src(sched, tensor, src_backend_id);
                if (non_weight_src_backend_id != -1) {
                    SET_CAUSE(tensor, "1.off-src%d", non_weight_src_backend_id);
                    return non_weight_src_backend_id;
                }
                for (int b = 0; b < src_backend_id; b++) {
                    if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                        SET_CAUSE(tensor, "1.off");
                        return b;
                    }
                }
            }
            SET_CAUSE(tensor, "1.wgt%d", i);
            return src_backend_id;
        }
    }

    return -1;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        snprintf(buffer, sizeof(buffer), "%zuM", size/1024/1024);
    } else {
        snprintf(buffer, sizeof(buffer), "%zuK", size/1024);
    }
    return buffer;
}

static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    int cur_split = 0;
    int active_split = -1;
    int gpu_backend_id = -1;
    if (sched->debug > 2) {
        for (int backend_id = 0; backend_id < sched->n_backends; ++backend_id) {
            if (ggml_backend_dev_type(ggml_backend_get_device(sched->backends[backend_id])) ==
                    GGML_BACKEND_DEVICE_TYPE_GPU) {
                gpu_backend_id = backend_id;
                break;
            }
        }
    }
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            active_split = cur_split;
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                if (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (sched->debug > 2) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
            ggml_backend_buffer_t node_buffer = node->view_src != nullptr ? node->view_src->buffer : node->buffer;
            const bool gpu_supported = gpu_backend_id >= 0 &&
                ggml_backend_supports_op(sched->backends[gpu_backend_id], node);
            const bool gpu_offload = gpu_backend_id >= 0 &&
                ggml_backend_offload_op(sched->backends[gpu_backend_id], node);
            GGML_LOG_DEBUG(
                "## NODE_FULL\t%llu\t%d\t%d\t%s\t%s\t%s\t%d\t%d\t%zu\t%lld\t%u\t%d\t%s\t%s\n",
                (unsigned long long) graph->uid, i, active_split,
                tensor_backend != nullptr ? ggml_backend_name(tensor_backend) : "NULL",
                GET_CAUSE(node), ggml_op_name(node->op), gpu_supported ? 1 : 0, gpu_offload ? 1 : 0,
                ggml_nbytes(node), (long long) ggml_nelements(node), (unsigned) node->flags,
                ggml_is_view_op(node->op) ? 1 : 0,
                node_buffer != nullptr ? ggml_backend_buffer_name(node_buffer) : "",
                node->name);
            for (int src_index = 0; src_index < GGML_MAX_SRC; ++src_index) {
                ggml_tensor * src = node->src[src_index];
                if (src == nullptr) {
                    continue;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                ggml_backend_buffer_t src_buffer = src->view_src != nullptr ? src->view_src->buffer : src->buffer;
                GGML_LOG_DEBUG(
                    "## SRC_FULL\t%llu\t%d\t%d\t%s\t%s\t%zu\t%lld\t%u\t%d\t%s\t%s\n",
                    (unsigned long long) graph->uid, i, src_index,
                    src_backend != nullptr ? ggml_backend_name(src_backend) : "NULL",
                    GET_CAUSE(src), ggml_nbytes(src), (long long) ggml_nelements(src), (unsigned) src->flags,
                    ggml_is_view_op(src->op) ? 1 : 0,
                    src_buffer != nullptr ? ggml_backend_buffer_name(src_buffer) : "",
                    src->name);
            }
        }
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        if (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_name(node->op), node->name,
                fmt_size(ggml_nbytes(node)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, node)], node->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

static int ggml_backend_sched_backend_from_non_weight_src(
        ggml_backend_sched_t sched,
        ggml_tensor * tensor,
        int max_backend_id) {
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        ggml_tensor * src = tensor->src[i];
        if (src == nullptr) {
            continue;
        }
        ggml_backend_buffer_t src_buffer = src->view_src != nullptr ? src->view_src->buffer : src->buffer;
        if (src_buffer != nullptr && src_buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            continue;
        }

        int src_backend_id = tensor_backend_id(src);
        if (src_backend_id == -1 && src->view_src != nullptr) {
            src_backend_id = tensor_backend_id(src->view_src);
        }
        if (src_backend_id < 0 || src_backend_id >= max_backend_id) {
            continue;
        }
        if (ggml_backend_supports_op(sched->backends[src_backend_id], tensor) &&
                ggml_backend_offload_op(sched->backends[src_backend_id], tensor)) {
            return src_backend_id;
        }
    }
    return -1;
}

static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;

    if (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        int tensor_backend_id = tensor_backend_id(t);
        if (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        if (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    if (sched->ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        // do not overwrite user assignments
        if (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);

#if 0
            // src
            if (node->op == GGML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < sched->n_backends; b++) {
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < *node_backend_id; b++) {
                if (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    bool supported = true;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        int i_split = 0;
        struct ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                break;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        int cur_backend_id = split->backend_id;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);

            GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // a weight homed on the drive, or a reshape of one (a view: it has no buffer of its own yet), is
                    // copied in a split of its own too: one copy target at a time
                    if (ggml_backend_sched_drive_home(sched, src) != nullptr) {
                        need_new_split = true;
                        break;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = true;
                            break;
                        }
                    }
                    // check if the split has too many inputs
                    // FIXME: count the number of inputs instead of only checking when full
                    if (split->n_inputs == GGML_SCHED_MAX_SPLIT_INPUTS) {
                        const size_t id = hash_id(src);
                        int src_backend_id = sched->hv_tensor_backend_ids[id];
                        bool supported = ggml_backend_sched_buffer_supported(sched, src, cur_backend_id);
                        if (src_backend_id != cur_backend_id && tensor_id_copy(id, cur_backend_id, 0) == NULL && !supported) {
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                if (i_split >= sched->splits_capacity) {
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = node_backend_id;
            }

            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                if (src->flags & GGML_TENSOR_FLAG_INPUT && sched->n_copies > 1) {
                    if (tensor_id_copy(src_id, src_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[src_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy;
                            if (c == sched->cur_copy) {
                                tensor_copy = src; // use the original tensor as the current copy
                            } else {
                                tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                                ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            }
                            ggml_set_input(tensor_copy);
                            ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            tensor_id_copy(src_id, src_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_graph_inputs = sched->n_graph_inputs++;
                        GGML_ASSERT(n_graph_inputs < GGML_SCHED_MAX_SPLIT_INPUTS);
                        sched->graph_inputs[n_graph_inputs] = src;
                    }
                }

                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_inputs = split->n_inputs++;
                        GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS);
                        split->inputs[n_inputs] = src;
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + sched->n_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sched->n_copies;

    // remember the actual graph_size for performing reallocation checks later [GGML_SCHED_DEBUG_REALLOC]
    sched->debug_prev_graph_size = sched->debug_graph_size;
    sched->debug_graph_size = graph_size;

    if (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * sizeof(struct ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * sizeof(struct ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    struct ggml_cgraph * graph_copy = &sched->graph;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        // Optimize this split of the graph. This needs to happen before we make graph_copy,
        // so they are in sync.
        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph);

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            struct ggml_tensor * input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];
        }
    }

    if (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < sched->n_copies; c++) {
                struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                assert(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    assert(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        assert(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    for (int i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        if (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            const bool unexpected = !backend_ids_changed && sched->debug_prev_graph_size == sched->debug_graph_size;

            if (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->debug_graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids);
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    return true;
}

static bool ggml_backend_sched_moe_log_enabled() {
    static const bool enabled = []() {
        const char * env = getenv("GGML_SCHED_MOE_LOG");
        return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0;
    }();
    return enabled;
}

static int ggml_backend_sched_moe_cache_slots() {
    static const int slots = []() {
        const char * env = getenv("GGML_SCHED_MOE_CACHE_SLOTS");
        if (env == nullptr || env[0] == '\0') {
            return 0;
        }

        errno = 0;
        char * end = nullptr;
        const long value = strtol(env, &end, 10);
        if (errno != 0 || end == env || *end != '\0' || value < 0 || value > INT_MAX) {
            GGML_LOG_WARN("%s: ignoring invalid GGML_SCHED_MOE_CACHE_SLOTS=%s\n", __func__, env);
            return 0;
        }

        return (int) value;
    }();
    return slots;
}

static enum ggml_backend_sched_moe_prefetch_policy ggml_backend_sched_moe_prefetch_policy() {
    static const enum ggml_backend_sched_moe_prefetch_policy policy = []() {
        const char * env = getenv("GGML_SCHED_MOE_CACHE_PREFETCH");
        if (env == nullptr || env[0] == '\0' || strcmp(env, "0") == 0 || strcmp(env, "none") == 0) {
            return GGML_BACKEND_SCHED_MOE_PREFETCH_NONE;
        }
        if (strcmp(env, "setmarkov") == 0) {
            return GGML_BACKEND_SCHED_MOE_PREFETCH_SETMARKOV;
        }

        GGML_LOG_WARN("%s: ignoring invalid GGML_SCHED_MOE_CACHE_PREFETCH=%s\n", __func__, env);
        return GGML_BACKEND_SCHED_MOE_PREFETCH_NONE;
    }();
    return policy;
}

static int ggml_backend_sched_moe_prefetch_limit() {
    static const int limit = []() {
        const char * env = getenv("GGML_SCHED_MOE_CACHE_PREFETCH_LIMIT");
        if (env == nullptr || env[0] == '\0') {
            return 0;
        }

        errno = 0;
        char * end = nullptr;
        const long value = strtol(env, &end, 10);
        if (errno != 0 || end == env || *end != '\0' || value < 0 || value > INT_MAX) {
            GGML_LOG_WARN("%s: ignoring invalid GGML_SCHED_MOE_CACHE_PREFETCH_LIMIT=%s\n", __func__, env);
            return 0;
        }

        return (int) value;
    }();
    return limit;
}

static bool ggml_backend_sched_moe_cache_prime_last_enabled() {
    static const bool enabled = []() {
        const char * env = getenv("GGML_SCHED_MOE_CACHE_PRIME");
        return env != nullptr && strcmp(env, "last") == 0;
    }();
    return enabled;
}

struct ggml_kimi_split_profile_key {
    int phase = 0;
    std::string backend;
    std::string first;
    std::string last;
    int i_start = 0;
    int i_end = 0;

    bool operator<(const ggml_kimi_split_profile_key & other) const {
        return std::tie(phase, backend, first, last, i_start, i_end) <
               std::tie(other.phase, other.backend, other.first, other.last, other.i_start, other.i_end);
    }
};

struct ggml_kimi_split_profile_value {
    uint64_t calls = 0;
    uint64_t nodes = 0;
    uint64_t wall_us = 0;
};

static std::mutex g_kimi_split_profile_mutex;
static std::map<ggml_kimi_split_profile_key, ggml_kimi_split_profile_value> g_kimi_split_profile;
static int g_kimi_split_profile_phase = 0;

struct ggml_kimi_layer_gap_sched_entry {
    uint64_t graph_compute_index = 0;
    int phase = 0;
    int split_id = 0;
    int n_nodes = 0;
    uint64_t split_start_us = 0;
    uint64_t inputs_end_us = 0;
    uint64_t compute_start_us = 0;
    uint64_t compute_end_us = 0;
    uint64_t event_end_us = 0;
    uint64_t split_end_us = 0;
    uint64_t moe_cache_us = 0;
    char backend[32] = {};
    char first[GGML_MAX_NAME] = {};
    char last[GGML_MAX_NAME] = {};
};

struct ggml_kimi_layer_gap_sched_buffer {
    std::vector<ggml_kimi_layer_gap_sched_entry> entries;
};

static const char * const g_kimi_layer_gap_trace_prefix =
    std::getenv("GGML_MOE_TRACE_LAYER_GAPS");
static const bool g_kimi_layer_gap_trace_enabled =
    g_kimi_layer_gap_trace_prefix != nullptr && g_kimi_layer_gap_trace_prefix[0] != '\0';
static std::atomic<uint64_t> g_kimi_layer_gap_graph_compute_index{0};
static std::once_flag g_kimi_layer_gap_sched_init_once;

static std::mutex & ggml_kimi_layer_gap_sched_registry_mutex() {
    static auto * mutex = new std::mutex();
    return *mutex;
}

static std::vector<ggml_kimi_layer_gap_sched_buffer *> & ggml_kimi_layer_gap_sched_registry() {
    static auto * registry = new std::vector<ggml_kimi_layer_gap_sched_buffer *>();
    return *registry;
}

static void ggml_kimi_layer_gap_copy_field(char * dst, size_t dst_size, const char * src) {
    if (dst_size == 0) {
        return;
    }
    size_t out = 0;
    for (const char * p = src != nullptr ? src : ""; *p != '\0' && out + 1 < dst_size; ++p) {
        const char ch = *p;
        dst[out++] = ch == ',' || ch == '\n' || ch == '\r' ? ' ' : ch;
    }
    dst[out] = '\0';
}

static void ggml_kimi_layer_gap_sched_report_atexit() {
    if (!g_kimi_layer_gap_trace_enabled) {
        return;
    }
    const std::string path = std::string(g_kimi_layer_gap_trace_prefix) + ".sched.csv";
    FILE * file = std::fopen(path.c_str(), "w");
    if (file == nullptr) {
        std::fprintf(stderr, "[kimi_layer_gap] scheduler trace open failed: %s\n", path.c_str());
        return;
    }
    std::fprintf(
        file,
        "graph_compute_index,phase,split_id,backend,n_nodes,first,last,"
        "split_start_us,inputs_end_us,compute_start_us,compute_end_us,"
        "event_end_us,split_end_us,moe_cache_us\n");
    size_t rows = 0;
    {
        std::lock_guard<std::mutex> lock(ggml_kimi_layer_gap_sched_registry_mutex());
        for (const ggml_kimi_layer_gap_sched_buffer * buffer : ggml_kimi_layer_gap_sched_registry()) {
            for (const ggml_kimi_layer_gap_sched_entry & entry : buffer->entries) {
                std::fprintf(
                    file,
                    "%" PRIu64 ",%d,%d,%s,%d,%s,%s,"
                    "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ","
                    "%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
                    entry.graph_compute_index,
                    entry.phase,
                    entry.split_id,
                    entry.backend,
                    entry.n_nodes,
                    entry.first,
                    entry.last,
                    entry.split_start_us,
                    entry.inputs_end_us,
                    entry.compute_start_us,
                    entry.compute_end_us,
                    entry.event_end_us,
                    entry.split_end_us,
                    entry.moe_cache_us);
                ++rows;
            }
        }
    }
    std::fclose(file);
    std::fprintf(stderr, "[kimi_layer_gap] scheduler trace written: %s (%zu rows)\n", path.c_str(), rows);
}

static ggml_kimi_layer_gap_sched_buffer & ggml_kimi_layer_gap_sched_thread_buffer() {
    static thread_local ggml_kimi_layer_gap_sched_buffer * buffer = nullptr;
    if (buffer == nullptr) {
        std::call_once(g_kimi_layer_gap_sched_init_once, []() {
            std::atexit(ggml_kimi_layer_gap_sched_report_atexit);
            std::fprintf(
                stderr,
                "[kimi_layer_gap] scheduler trace enabled: prefix=%s\n",
                g_kimi_layer_gap_trace_prefix);
        });
        buffer = new ggml_kimi_layer_gap_sched_buffer();
        buffer->entries.reserve(1U << 14);
        std::lock_guard<std::mutex> lock(ggml_kimi_layer_gap_sched_registry_mutex());
        ggml_kimi_layer_gap_sched_registry().push_back(buffer);
    }
    return *buffer;
}

void ggml_backend_sched_kimi_set_profile_phase(int phase) {
    g_kimi_split_profile_phase = phase >= 0 && phase <= 2 ? phase : 0;
}

void ggml_backend_sched_kimi_set_dense_decode(ggml_backend_sched_t sched, bool decode) {
    GGML_ASSERT(sched != nullptr);
    sched->decode_phase = decode;
}

static int ggml_kimi_split_profile_top() {
    const char * env = getenv("GGML_KIMI_SPLIT_PROFILE_TOP");
    int top = (env && env[0]) ? atoi(env) : 40;
    if (top < 1) {
        top = 1;
    }
    if (top > 512) {
        top = 512;
    }
    return top;
}

static void ggml_kimi_split_profile_report() {
    std::vector<std::pair<ggml_kimi_split_profile_key, ggml_kimi_split_profile_value>> rows;
    {
        std::lock_guard<std::mutex> lock(g_kimi_split_profile_mutex);
        rows.reserve(g_kimi_split_profile.size());
        for (const auto & kv : g_kimi_split_profile) {
            rows.push_back(kv);
        }
    }
    if (rows.empty()) {
        return;
    }
    std::sort(rows.begin(), rows.end(), [](const auto & a, const auto & b) {
        return a.second.wall_us > b.second.wall_us;
    });
    auto report_phase = [&](const char * label, int phase_filter) {
        uint64_t total_us = 0;
        uint64_t total_calls = 0;
        size_t signatures = 0;
        std::vector<size_t> indexes;
        std::map<std::string, ggml_kimi_split_profile_value> backend_totals;
        indexes.reserve(rows.size());
        for (size_t i = 0; i < rows.size(); ++i) {
            if (phase_filter >= 0 && rows[i].first.phase != phase_filter) {
                continue;
            }
            const auto & key = rows[i].first;
            const auto & val = rows[i].second;
            total_us += val.wall_us;
            total_calls += val.calls;
            signatures++;
            indexes.push_back(i);
            ggml_kimi_split_profile_value & backend_val = backend_totals[key.backend];
            backend_val.calls += val.calls;
            backend_val.nodes += val.nodes;
            backend_val.wall_us += val.wall_us;
        }
        if (signatures == 0) {
            return;
        }
        GGML_LOG_INFO(
            "[kimi_split_profile] phase=%s total: signatures=%zu calls=%" PRIu64 " wall=%.3f ms\n",
            label, signatures, total_calls, (double) total_us / 1000.0);
        for (const auto & backend_kv : backend_totals) {
            const auto & backend = backend_kv.first;
            const auto & val = backend_kv.second;
            GGML_LOG_INFO(
                "[kimi_split_profile] phase=%s backend=%s total: calls=%" PRIu64
                " nodes=%" PRIu64 " wall=%.3f ms avg=%.3f ms/call\n",
                label,
                backend.c_str(),
                val.calls,
                val.nodes,
                (double) val.wall_us / 1000.0,
                val.calls == 0 ? 0.0 : (double) val.wall_us / 1000.0 / (double) val.calls);
        }
        const int top = std::min<int>(ggml_kimi_split_profile_top(), (int) indexes.size());
        for (int i = 0; i < top; ++i) {
            const auto & row = rows[indexes[i]];
            const auto & key = row.first;
            const auto & val = row.second;
            GGML_LOG_INFO(
                "[kimi_split_profile] phase=%s top%d: backend=%s range=%d:%d nodes_avg=%.2f calls=%" PRIu64
                " wall=%.3f ms avg=%.3f ms/call first=%s last=%s\n",
                label,
                i + 1,
                key.backend.c_str(),
                key.i_start,
                key.i_end,
                val.calls == 0 ? 0.0 : (double) val.nodes / (double) val.calls,
                val.calls,
                (double) val.wall_us / 1000.0,
                val.calls == 0 ? 0.0 : (double) val.wall_us / 1000.0 / (double) val.calls,
                key.first.c_str(),
                key.last.c_str());
        }
    };
    report_phase("all", -1);
    report_phase("prompt", 1);
    report_phase("decode", 2);
    report_phase("idle", 0);
}

static bool ggml_kimi_split_profile_enabled() {
    static bool enabled = []() {
        const char * env = getenv("GGML_KIMI_SPLIT_PROFILE");
        const bool on = env && env[0] && env[0] != '0';
        if (on) {
            atexit(ggml_kimi_split_profile_report);
        }
        return on;
    }();
    return enabled;
}

struct ggml_kimi_sched_pack_source_profile {
    uint64_t ranges_attempted = 0;
    uint64_t ranges_replaced = 0;
    uint64_t ranges_fallback = 0;
    uint64_t experts_attempted = 0;
    uint64_t experts_read = 0;
    uint64_t bytes_replaced = 0;
    uint64_t gguf_source_bytes = 0;
    uint64_t read_us = 0;
    uint64_t syncs = 0;
    uint64_t sync_us = 0;
    uint64_t batch_calls = 0;
    uint64_t batch_success = 0;
    uint64_t batch_fallback = 0;
    uint64_t iouring_reads = 0;
    uint64_t iouring_bytes = 0;
    uint64_t cache_lookups = 0;
    uint64_t cache_hits = 0;
    uint64_t cache_bypass_experts = 0;
};

static std::mutex g_kimi_sched_pack_source_profile_mutex;
static ggml_kimi_sched_pack_source_profile g_kimi_sched_pack_source_profile;

// the copies of the weights homed on the drive (ggml_backend_sched_drive_home), under the profile's mutex
struct ggml_kimi_sched_dense_stream {
    uint64_t tensors = 0;
    uint64_t bytes = 0;
    uint64_t read_us = 0;           // in the transport's copy: the reads not done yet, the device copies queued
    uint64_t first_us = 0;          // of it, in the copies whose blocks had not been read ahead
    uint64_t wait_us = 0;           // in the scheduler's wait for the device before such a copy
    uint64_t readahead = 0;         // copies whose blocks had been read ahead
    uint64_t fallbacks = 0;         // copies the transport did not make: the mapped weight copied instead
};

static ggml_kimi_sched_dense_stream g_kimi_sched_dense_stream;

static bool ggml_kimi_env_enabled(const char * name) {
    const char * env = getenv(name);
    return env && env[0] && env[0] != '0';
}

static const char * ggml_kimi_sched_pack_source_profile_path() {
    static const char * path = []() -> const char * {
        const char * env = getenv("GGML_SCHED_MOE_COPY_EXPERT_PACK_SOURCE_PROFILE_OUT");
        return env && env[0] ? env : nullptr;
    }();
    return path;
}

static void ggml_kimi_sched_pack_source_profile_report() {
    const char * path = ggml_kimi_sched_pack_source_profile_path();
    if (!path) {
        return;
    }
    ggml_kimi_sched_pack_source_profile p;
    {
        std::lock_guard<std::mutex> lock(g_kimi_sched_pack_source_profile_mutex);
        p = g_kimi_sched_pack_source_profile;
    }
    FILE * f = fopen(path, "w");
    if (!f) {
        return;
    }
    fprintf(f, "ranges_attempted,ranges_replaced,ranges_fallback,experts_attempted,experts_read,bytes_replaced,gguf_source_bytes,read_us,syncs,sync_us,batch_calls,batch_success,batch_fallback,iouring_reads,iouring_bytes,cache_lookups,cache_hits,cache_bypass_experts\n");
    fprintf(f, "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
            p.ranges_attempted,
            p.ranges_replaced,
            p.ranges_fallback,
            p.experts_attempted,
            p.experts_read,
            p.bytes_replaced,
            p.gguf_source_bytes,
            p.read_us,
            p.syncs,
            p.sync_us,
            p.batch_calls,
            p.batch_success,
            p.batch_fallback,
            p.iouring_reads,
            p.iouring_bytes,
            p.cache_lookups,
            p.cache_hits,
            p.cache_bypass_experts);
    fclose(f);
}

void ggml_backend_sched_kimi_copy_stats_get(struct ggml_backend_sched_kimi_copy_stats * stats) {
    std::lock_guard<std::mutex> lock(g_kimi_sched_pack_source_profile_mutex);
    const auto & p = g_kimi_sched_pack_source_profile;
    stats->expert_slices = p.experts_read;
    stats->bytes = p.bytes_replaced;
    stats->read_us = p.read_us;
    stats->fallbacks = p.batch_fallback + p.ranges_fallback;
    const auto & d = g_kimi_sched_dense_stream;
    stats->dense_tensors = d.tensors;
    stats->dense_bytes = d.bytes;
    stats->dense_read_us = d.read_us;
    stats->dense_first_us = d.first_us;
    stats->dense_wait_us = d.wait_us;
    stats->dense_readahead = d.readahead;
    stats->dense_fallbacks = d.fallbacks;
}

static bool ggml_kimi_sched_pack_source_enabled() {
    static const bool enabled = []() {
        const bool on = ggml_kimi_env_enabled("GGML_SCHED_MOE_COPY_EXPERT_PACK_SOURCE");
        if (on && ggml_kimi_sched_pack_source_profile_path()) {
            atexit(ggml_kimi_sched_pack_source_profile_report);
        }
        return on;
    }();
    return enabled;
}

static bool ggml_kimi_sched_pack_source_prompt_only() {
    static const bool enabled = []() {
        const char * env = getenv("GGML_SCHED_MOE_COPY_EXPERT_PACK_SOURCE_PROMPT_ONLY");
        return !(env && env[0] && env[0] == '0');
    }();
    return enabled;
}

static bool ggml_kimi_sched_pack_source_batch_enabled() {
    static const bool enabled = []() {
        const char * env = getenv("GGML_SCHED_MOE_COPY_EXPERT_PACK_SOURCE_BATCH");
        return !(env && env[0] && env[0] == '0');
    }();
    return enabled;
}

static bool ggml_kimi_sched_pack_source_cache_bypass() {
    static const bool enabled = []() {
        const char * env = getenv("GGML_SCHED_MOE_COPY_EXPERT_PACK_SOURCE_CACHE_BYPASS");
        return !(env && env[0] && env[0] == '0');
    }();
    return enabled;
}

static bool ggml_kimi_sched_pack_source_symbol_available() {
    return ggml_cuda_moe_expert_pack_copy_ranges_to_tensor_resolve() != nullptr ||
        ggml_cuda_moe_expert_pack_read_to_host_resolve() != nullptr;
}

static bool ggml_kimi_sched_pack_source_read_range(
        const char * tensor_name,
        int32_t first_id,
        int32_t last_id,
        size_t expert_size,
        size_t padding_end,
        std::vector<std::vector<uint8_t>> & buffers,
        const void ** out_ptr,
        size_t * out_bytes) {
#if defined(__GNUC__) || defined(__clang__)
    const auto read_to_host = ggml_cuda_moe_expert_pack_read_to_host_resolve();
    if (!read_to_host || !tensor_name || !tensor_name[0] ||
            first_id < 0 || last_id < first_id || expert_size == 0) {
        return false;
    }

    const size_t n_experts = (size_t) (last_id - first_id + 1);
    const size_t payload_bytes = n_experts * expert_size;
    const size_t bytes = payload_bytes + padding_end;
    std::vector<uint8_t> buffer(bytes);
    if (padding_end > 0) {
        memset(buffer.data() + payload_bytes, 0, padding_end);
    }

    const int64_t t_start_us = ggml_time_us();
    uint64_t experts_read = 0;
    uint64_t gguf_source_bytes = 0;
    bool ok = true;
    for (int32_t expert = first_id; expert <= last_id; ++expert) {
        size_t nread = 0;
        int source_is_gguf = 0;
        void * dst = buffer.data() + (size_t) (expert - first_id) * expert_size;
        if (!read_to_host(
                    tensor_name, expert, expert_size, dst, expert_size, &nread, &source_is_gguf) ||
                nread != expert_size) {
            ok = false;
            break;
        }
        ++experts_read;
        if (source_is_gguf) {
            gguf_source_bytes += expert_size;
        }
    }
    const uint64_t read_us = (uint64_t) (ggml_time_us() - t_start_us);

    {
        std::lock_guard<std::mutex> lock(g_kimi_sched_pack_source_profile_mutex);
        g_kimi_sched_pack_source_profile.ranges_attempted++;
        g_kimi_sched_pack_source_profile.experts_attempted += n_experts;
        g_kimi_sched_pack_source_profile.experts_read += experts_read;
        g_kimi_sched_pack_source_profile.read_us += read_us;
        if (ok) {
            g_kimi_sched_pack_source_profile.ranges_replaced++;
            g_kimi_sched_pack_source_profile.bytes_replaced += bytes;
            g_kimi_sched_pack_source_profile.gguf_source_bytes += gguf_source_bytes;
        } else {
            g_kimi_sched_pack_source_profile.ranges_fallback++;
        }
    }

    if (!ok) {
        return false;
    }
    buffers.emplace_back(std::move(buffer));
    *out_ptr = buffers.back().data();
    *out_bytes = bytes;
    return true;
#else
    (void) tensor_name;
    (void) first_id;
    (void) last_id;
    (void) expert_size;
    (void) padding_end;
    (void) buffers;
    (void) out_ptr;
    (void) out_bytes;
    return false;
#endif
}

static void ggml_kimi_sched_pack_source_profile_record_sync(uint64_t sync_us) {
    std::lock_guard<std::mutex> lock(g_kimi_sched_pack_source_profile_mutex);
    g_kimi_sched_pack_source_profile.syncs++;
    g_kimi_sched_pack_source_profile.sync_us += sync_us;
}

static void ggml_kimi_sched_pack_source_profile_record_batch(
        bool ok,
        size_t ranges,
        uint64_t experts,
        uint64_t bytes,
        uint64_t read_us,
        uint64_t iouring_reads,
        uint64_t iouring_bytes,
        uint64_t cache_lookups,
        uint64_t cache_hits,
        bool cache_bypass) {
    std::lock_guard<std::mutex> lock(g_kimi_sched_pack_source_profile_mutex);
    auto & p = g_kimi_sched_pack_source_profile;
    ++p.batch_calls;
    p.iouring_reads += iouring_reads;
    p.iouring_bytes += iouring_bytes;
    p.cache_lookups += cache_lookups;
    p.cache_hits += cache_hits;
    if (!ok) {
        ++p.batch_fallback;
        return;
    }
    ++p.batch_success;
    p.ranges_attempted += ranges;
    p.ranges_replaced += ranges;
    p.experts_attempted += experts;
    p.experts_read += experts;
    p.bytes_replaced += bytes;
    p.read_us += read_us;
    if (cache_bypass) p.cache_bypass_experts += experts;
}

static bool ggml_kimi_split_moe_assign_profile_enabled() {
    static const bool enabled = []() {
        const char * env = getenv("GGML_KIMI_SPLIT_MOE_ASSIGN_PROFILE");
        return env && env[0] && env[0] != '0';
    }();
    return enabled;
}

static int ggml_kimi_split_moe_assign_profile_limit() {
    static const int limit = []() {
        const char * env = getenv("GGML_KIMI_SPLIT_MOE_ASSIGN_PROFILE_LIMIT");
        int value = (env && env[0]) ? atoi(env) : 256;
        if (value < 1) {
            value = 1;
        }
        if (value > 4096) {
            value = 4096;
        }
        return value;
    }();
    return limit;
}

static bool ggml_kimi_name_contains(const char * name, const char * needle) {
    return name != nullptr && needle != nullptr && strstr(name, needle) != nullptr;
}

static const char * ggml_kimi_tensor_buffer_name(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return "<null>";
    }
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    return buffer != nullptr ? ggml_backend_buffer_name(buffer) : "<none>";
}

static int ggml_kimi_tensor_buffer_usage(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return -1;
    }
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    return buffer != nullptr ? (int) buffer->usage : -1;
}

static bool ggml_kimi_tensor_buffer_name_contains(const ggml_tensor * tensor, const char * needle) {
    return strstr(ggml_kimi_tensor_buffer_name(tensor), needle) != nullptr;
}

static void ggml_kimi_split_moe_assign_profile_record(
        ggml_backend_sched_t sched,
        int split_id,
        const ggml_backend_sched_split * split,
        ggml_backend_t split_backend) {
    const bool assign_profile = ggml_kimi_split_moe_assign_profile_enabled();
    if (!assign_profile ||
            g_kimi_split_profile_phase != 2 ||
            sched == nullptr || split == nullptr || split_backend == nullptr ||
            split->graph.n_nodes <= 0) {
        return;
    }
    if (strstr(ggml_backend_name(split_backend), "CPU") == nullptr) {
        return;
    }
    const ggml_tensor * first = split->graph.nodes[0];
    const ggml_tensor * last = split->graph.nodes[split->graph.n_nodes - 1];
    const char * first_name = first && first->name[0] ? first->name : "<unnamed>";
    const char * last_name = last && last->name[0] ? last->name : "<unnamed>";
    const bool moe_split =
        ggml_kimi_name_contains(first_name, "ffn_moe_swiglu") ||
        ggml_kimi_name_contains(first_name, "ffn_moe_down") ||
        ggml_kimi_name_contains(last_name, "ffn_moe_swiglu") ||
        ggml_kimi_name_contains(last_name, "ffn_moe_down");
    if (!moe_split) {
        return;
    }

    static int emitted = 0;
    if (emitted >= ggml_kimi_split_moe_assign_profile_limit()) {
        return;
    }
    emitted++;

    GGML_LOG_INFO(
        "[kimi_moe_assign_profile] split=%d backend=%s range=%d:%d nodes=%d first=%s last=%s\n",
        split_id,
        ggml_backend_name(split_backend),
        split->i_start,
        split->i_end,
        split->graph.n_nodes,
        first_name,
        last_name);

    for (int node_id = 0; node_id < split->graph.n_nodes; ++node_id) {
        const ggml_tensor * node = split->graph.nodes[node_id];
        std::string support;
        for (int b = 0; b < sched->n_backends; ++b) {
            if (!support.empty()) {
                support += ",";
            }
            support += ggml_backend_name(sched->backends[b]);
            support += ":";
            support += ggml_backend_supports_op(sched->backends[b], node) ? "1" : "0";
        }
        GGML_LOG_INFO(
            "[kimi_moe_assign_profile] split=%d node=%d name=%s op=%s type=%s buf=%s usage=%d supports=[%s]\n",
            split_id,
            node_id,
            node && node->name[0] ? node->name : "<unnamed>",
            node ? ggml_op_name(node->op) : "<null>",
            node ? ggml_type_name(node->type) : "<null>",
            ggml_kimi_tensor_buffer_name(node),
            ggml_kimi_tensor_buffer_usage(node),
            support.c_str());
        if (node == nullptr) {
            continue;
        }
        for (int src_id = 0; src_id < GGML_MAX_SRC; ++src_id) {
            const ggml_tensor * src = node->src[src_id];
            if (src == nullptr) {
                continue;
            }
            GGML_LOG_INFO(
                "[kimi_moe_assign_profile] split=%d node=%d src=%d name=%s op=%s type=%s buf=%s usage=%d\n",
                split_id,
                node_id,
                src_id,
                src->name[0] ? src->name : "<unnamed>",
                ggml_op_name(src->op),
                ggml_type_name(src->type),
                ggml_kimi_tensor_buffer_name(src),
                ggml_kimi_tensor_buffer_usage(src));
        }
    }
}

static void ggml_kimi_split_profile_record(
        const ggml_backend_sched_split * split,
        ggml_backend_t backend,
        uint64_t wall_us) {
    if (split == nullptr || backend == nullptr || split->graph.n_nodes <= 0) {
        return;
    }
    ggml_kimi_split_profile_key key;
    key.phase = g_kimi_split_profile_phase;
    key.backend = ggml_backend_name(backend);
    key.i_start = split->i_start;
    key.i_end = split->i_end;
    const ggml_tensor * first = split->graph.nodes[0];
    const ggml_tensor * last = split->graph.nodes[split->graph.n_nodes - 1];
    key.first = first && first->name[0] ? first->name : "<unnamed>";
    key.last = last && last->name[0] ? last->name : "<unnamed>";
    std::lock_guard<std::mutex> lock(g_kimi_split_profile_mutex);
    ggml_kimi_split_profile_value & val = g_kimi_split_profile[key];
    val.calls++;
    val.nodes += (uint64_t) split->graph.n_nodes;
    val.wall_us += wall_us;
}

static const char * ggml_backend_sched_tensor_name(const ggml_tensor * tensor) {
    return tensor->name[0] != '\0' ? tensor->name : "<unnamed>";
}

static bool ggml_backend_sched_same_layout(const ggml_tensor * a, const ggml_tensor * b) {
    if (a->type != b->type) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (a->ne[i] != b->ne[i] || a->nb[i] != b->nb[i]) {
            return false;
        }
    }
    return true;
}

static ggml_backend_sched_moe_cache * ggml_backend_sched_moe_cache_find(
        ggml_backend_sched_t sched,
        const ggml_tensor * input,
        int backend_id) {
    for (ggml_backend_sched_moe_cache * cache : *sched->moe_caches) {
        if (cache->input == input && cache->backend_id == backend_id) {
            return cache;
        }
    }
    return nullptr;
}

static size_t ggml_backend_sched_moe_cache_slot_padding(const ggml_tensor * input, size_t expert_size) {
    const size_t type_size = ggml_type_size(input->type);
    GGML_ASSERT(type_size > 0);
    GGML_ASSERT(expert_size % type_size == 0);

    const size_t padding = std::min<size_t>(expert_size, 512);
    return ((padding + type_size - 1) / type_size) * type_size;
}

static ggml_backend_sched_moe_cache * ggml_backend_sched_moe_cache_new(
        ggml_backend_sched_t sched,
        ggml_backend_t backend,
        const ggml_tensor * input,
        int backend_id,
        int n_expert,
        int n_slots,
        size_t expert_size) {
    GGML_ASSERT(n_slots > 0);
    GGML_ASSERT(n_slots <= n_expert);
    GGML_ASSERT(input->ne[3] == 1);

    ggml_backend_buffer_type_t buft = sched->bufts[backend_id];
    const size_t padding = ggml_backend_sched_moe_cache_slot_padding(input, expert_size);

    ggml_backend_sched_moe_cache * cache = new ggml_backend_sched_moe_cache();
    cache->input       = input;
    cache->backend_id  = backend_id;
    cache->n_expert    = n_expert;
    cache->n_slots     = n_slots;
    cache->expert_size = expert_size;
    cache->slot_padding = padding;
    cache->slot_stride = expert_size + padding;
    cache->weights_size = 0;

    cache->weights_tensor = *input;
    cache->weights_tensor.buffer = nullptr;
    cache->weights_tensor.data = nullptr;
    cache->weights_tensor.view_src = nullptr;
    cache->weights_tensor.view_offs = 0;
    cache->weights_tensor.extra = nullptr;
    cache->weights_tensor.op = GGML_OP_NONE;
    cache->weights_tensor.flags = 0;
    cache->weights_tensor.ne[2] = n_slots + 1; // one dummy padding slot
    cache->weights_tensor.ne[3] = 1;
    cache->weights_tensor.nb[2] = cache->slot_stride;
    cache->weights_tensor.nb[3] = cache->slot_stride * cache->weights_tensor.ne[2];
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        cache->weights_tensor.src[i] = nullptr;
    }
    ggml_format_name(&cache->weights_tensor, "%s#moe-cache#%s",
            ggml_backend_sched_tensor_name(input), ggml_backend_name(backend));

    const size_t weights_size = ggml_backend_buft_get_alloc_size(buft, &cache->weights_tensor);
    cache->weights_size = weights_size;
    cache->weights_buffer = ggml_backend_buft_alloc_buffer(buft, weights_size);
    if (cache->weights_buffer == nullptr) {
        delete cache;
        return nullptr;
    }
    ggml_backend_buffer_set_usage(cache->weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (ggml_backend_tensor_alloc(cache->weights_buffer, &cache->weights_tensor, ggml_backend_buffer_get_base(cache->weights_buffer)) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(cache->weights_buffer);
        delete cache;
        return nullptr;
    }
    ggml_backend_buffer_clear(cache->weights_buffer, 0);

    cache->slot_of.assign(n_expert, -1);
    cache->expert_in_slot.assign(n_slots, -1);
    cache->slot_speculative.assign(n_slots, 0);
    cache->lru_tick.assign(n_slots, 0);

    sched->moe_caches->push_back(cache);

    GGML_LOG_INFO("%s: allocated MoE expert cache for %s on %s: slots=%d/%d, bytes=%zu\n",
            __func__, ggml_backend_sched_tensor_name(input), ggml_backend_name(backend),
            n_slots, n_expert, weights_size);

    return cache;
}

static bool ggml_backend_sched_moe_cache_ensure_ids(
        ggml_backend_sched_moe_cache * cache,
        ggml_backend_buffer_type_t buft,
        const ggml_tensor * ids_tensor) {
    const size_t ids_nbytes = ggml_nbytes(ids_tensor);
    if (cache->ids_buffer != nullptr &&
            cache->ids_nbytes == ids_nbytes &&
            ggml_backend_sched_same_layout(&cache->ids_tensor, ids_tensor)) {
        return true;
    }

    ggml_backend_buffer_free(cache->ids_buffer);
    cache->ids_buffer = nullptr;
    cache->ids_nbytes = 0;

    cache->ids_tensor = *ids_tensor;
    cache->ids_tensor.buffer = nullptr;
    cache->ids_tensor.data = nullptr;
    cache->ids_tensor.view_src = nullptr;
    cache->ids_tensor.view_offs = 0;
    cache->ids_tensor.extra = nullptr;
    cache->ids_tensor.op = GGML_OP_NONE;
    cache->ids_tensor.flags = 0;
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        cache->ids_tensor.src[i] = nullptr;
    }
    ggml_format_name(&cache->ids_tensor, "%s#moe-cache-ids",
            ggml_backend_sched_tensor_name(ids_tensor));

    const size_t ids_alloc = ggml_backend_buft_get_alloc_size(buft, &cache->ids_tensor);
    cache->ids_buffer = ggml_backend_buft_alloc_buffer(buft, ids_alloc);
    if (cache->ids_buffer == nullptr) {
        return false;
    }
    ggml_backend_buffer_set_usage(cache->ids_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    if (ggml_backend_tensor_alloc(cache->ids_buffer, &cache->ids_tensor, ggml_backend_buffer_get_base(cache->ids_buffer)) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(cache->ids_buffer);
        cache->ids_buffer = nullptr;
        return false;
    }

    cache->ids_nbytes = ids_nbytes;
    return true;
}

static bool ggml_backend_sched_moe_cache_prepare(
        ggml_backend_sched_t sched,
        ggml_backend_t split_backend,
        int split_backend_id,
        ggml_tensor * input,
        ggml_tensor * node,
        ggml_tensor * ids_tensor,
        const std::vector<int32_t> & ids,
        const std::vector<ggml_bitset_t> & used_ids,
        int64_t n_expert,
        size_t expert_size,
        int requested_slots,
        enum ggml_backend_sched_moe_prefetch_policy prefetch_policy,
        int prefetch_limit,
        bool moe_log,
        const char ** fail_reason,
        std::vector<ggml_backend_sched_moe_restore> & restores) {
    if (fail_reason != nullptr) {
        *fail_reason = nullptr;
    }
    auto fail = [fail_reason](const char * reason) {
        if (fail_reason != nullptr) {
            *fail_reason = reason;
        }
        return false;
    };

    if (requested_slots <= 0) {
        return fail("disabled");
    }
    if (input->ne[3] != 1) {
        return fail("unsupported_shape");
    }
    if (n_expert <= 0 || n_expert > INT_MAX) {
        return fail("invalid_expert_count");
    }

    const int n_slots = std::min<int>(requested_slots, (int) n_expert);
    std::vector<int32_t> needed;
    needed.reserve(n_slots);
    for (int64_t i = 0; i < n_expert; ++i) {
        if (ggml_bitset_get(used_ids.data(), i)) {
            needed.push_back((int32_t) i);
        }
    }
    if (needed.empty()) {
        return fail("no_experts");
    }

    const bool too_many_experts = (int) needed.size() > n_slots;
    const bool prime_last = too_many_experts && ggml_backend_sched_moe_cache_prime_last_enabled();
    if (too_many_experts && !prime_last) {
        return fail("too_many_experts");
    }

    ggml_backend_sched_moe_cache * cache = ggml_backend_sched_moe_cache_find(sched, input, split_backend_id);
    if (cache != nullptr &&
            (cache->n_expert != n_expert || cache->n_slots != n_slots || cache->expert_size != expert_size)) {
        return fail("cache_metadata_mismatch");
    }
    if (cache == nullptr) {
        cache = ggml_backend_sched_moe_cache_new(sched, split_backend, input, split_backend_id, (int) n_expert, n_slots, expert_size);
        if (cache == nullptr) {
            return fail("cache_alloc_failed");
        }
    }

    if (too_many_experts) {
        std::vector<int32_t> prime_ids;
        prime_ids.reserve((size_t) n_slots);
        std::vector<uint8_t> seen((size_t) n_expert, 0);
        for (auto it = ids.rbegin(); it != ids.rend() && (int) prime_ids.size() < n_slots; ++it) {
            const int32_t expert_id = *it;
            if (expert_id < 0 || expert_id >= n_expert || seen[expert_id]) {
                continue;
            }
            seen[expert_id] = 1;
            prime_ids.push_back(expert_id);
        }
        std::reverse(prime_ids.begin(), prime_ids.end());

        size_t primed = 0;
        size_t primed_bytes = 0;
        for (int32_t expert_id : prime_ids) {
            int32_t slot = cache->slot_of[expert_id];
            if (slot >= 0) {
                cache->slot_speculative[slot] = 1;
                cache->lru_tick[slot] = ++cache->now;
                continue;
            }

            for (int32_t candidate = 0; candidate < cache->n_slots; ++candidate) {
                if (cache->expert_in_slot[candidate] == -1) {
                    slot = candidate;
                    break;
                }
            }

            if (slot == -1) {
                uint64_t best_tick = std::numeric_limits<uint64_t>::max();
                for (int32_t candidate = 0; candidate < cache->n_slots; ++candidate) {
                    if (cache->lru_tick[candidate] < best_tick) {
                        best_tick = cache->lru_tick[candidate];
                        slot = candidate;
                    }
                }
            }
            if (slot == -1) {
                continue;
            }

            const int32_t old_expert = cache->expert_in_slot[slot];
            if (old_expert >= 0) {
                if (cache->slot_speculative[slot]) {
                    cache->wrong_prefetches++;
                }
                cache->slot_of[old_expert] = -1;
            }

            const size_t padding = expert_id < n_expert - 1 ? cache->slot_padding : 0;
            const size_t copy_size = expert_size + padding;
            ggml_backend_tensor_set_async(split_backend,
                    &cache->weights_tensor,
                    (const uint8_t *) input->data + (size_t) expert_id * expert_size,
                    (size_t) slot * cache->slot_stride,
                    copy_size);

            cache->expert_in_slot[slot] = expert_id;
            cache->slot_of[expert_id] = slot;
            cache->slot_speculative[slot] = 1;
            cache->lru_tick[slot] = ++cache->now;
            cache->prefetches++;
            cache->bytes_prefetched += copy_size;
            primed++;
            primed_bytes += copy_size;
        }

        if (moe_log) {
            GGML_LOG_INFO("%s: moe_cache_prime tensor=%s backend=%s slots=%d primed=%zu primed_bytes=%zu used=%zu\n",
                    __func__, ggml_backend_sched_tensor_name(input), ggml_backend_name(split_backend),
                    n_slots, primed, primed_bytes, needed.size());
        }
        return fail("too_many_experts_primed");
    }

    if (!ggml_backend_sched_moe_cache_ensure_ids(cache, sched->bufts[split_backend_id], ids_tensor)) {
        return fail("ids_alloc_failed");
    }

    std::vector<ggml_bitset_t> predicted_ids;
    if (prefetch_policy == GGML_BACKEND_SCHED_MOE_PREFETCH_SETMARKOV && needed.size() <= 64) {
        ggml_backend_sched_moe_transition * prediction = nullptr;
        for (ggml_backend_sched_moe_transition & candidate : cache->transitions) {
            if (candidate.from == needed) {
                prediction = &candidate;
                break;
            }
        }

        if (prediction != nullptr) {
            std::vector<std::pair<int32_t, uint32_t>> candidates;
            candidates.reserve((size_t) n_expert);
            for (int64_t expert_id = 0; expert_id < n_expert; ++expert_id) {
                const uint32_t count = prediction->counts[expert_id];
                if (count > 0) {
                    candidates.push_back({ (int32_t) expert_id, count });
                }
            }
            std::sort(candidates.begin(), candidates.end(),
                    [](const std::pair<int32_t, uint32_t> & a, const std::pair<int32_t, uint32_t> & b) {
                if (a.second != b.second) {
                    return a.second > b.second;
                }
                return a.first < b.first;
            });

            const size_t protect_limit = std::min<size_t>((size_t) cache->n_slots, 6);
            predicted_ids.resize(ggml_bitset_size(n_expert));
            for (size_t i = 0; i < std::min(protect_limit, candidates.size()); ++i) {
                ggml_bitset_set(predicted_ids.data(), candidates[i].first);
            }
        }
    }

    std::vector<int32_t> misses;
    misses.reserve(needed.size());
    for (int32_t expert_id : needed) {
        const int32_t slot = cache->slot_of[expert_id];
        if (slot >= 0) {
            cache->hits++;
            if (cache->slot_speculative[slot]) {
                cache->speculative_hits++;
                cache->slot_speculative[slot] = 0;
            }
        } else {
            misses.push_back(expert_id);
            cache->misses++;
        }
    }

    auto find_free_slot = [&]() -> int32_t {
        for (int32_t slot = 0; slot < cache->n_slots; ++slot) {
            if (cache->expert_in_slot[slot] == -1) {
                return slot;
            }
        }
        return -1;
    };

    auto find_victim_slot = [&](bool prefer_speculative) -> int32_t {
        for (int pass = 0; pass < (prefer_speculative ? 4 : 2); ++pass) {
            const bool speculative_only = prefer_speculative && (pass % 2 == 0);
            const bool protect_predicted = pass < (prefer_speculative ? 2 : 1);
            int32_t slot = -1;
            uint64_t best_tick = std::numeric_limits<uint64_t>::max();
            for (int32_t candidate = 0; candidate < cache->n_slots; ++candidate) {
                const int32_t resident = cache->expert_in_slot[candidate];
                GGML_ASSERT(resident >= 0);
                if (ggml_bitset_get(used_ids.data(), resident)) {
                    continue;
                }
                if (protect_predicted && !predicted_ids.empty() && ggml_bitset_get(predicted_ids.data(), resident)) {
                    continue;
                }
                if (speculative_only && !cache->slot_speculative[candidate]) {
                    continue;
                }
                if (cache->lru_tick[candidate] < best_tick) {
                    best_tick = cache->lru_tick[candidate];
                    slot = candidate;
                }
            }
            if (slot >= 0) {
                return slot;
            }
        }
        return -1;
    };

    auto copy_expert_to_slot = [&](int32_t expert_id, int32_t slot, bool speculative, bool prefetch) -> size_t {
        const int32_t old_expert = cache->expert_in_slot[slot];
        if (old_expert >= 0) {
            if (cache->slot_speculative[slot]) {
                cache->wrong_prefetches++;
            }
            if (prefetch) {
                cache->prefetch_evictions++;
            }
            cache->slot_of[old_expert] = -1;
        }

        const size_t padding = expert_id < n_expert - 1 ? cache->slot_padding : 0;
        const size_t copy_size = expert_size + padding;
        ggml_backend_tensor_set_async(split_backend,
                &cache->weights_tensor,
                (const uint8_t *) input->data + (size_t) expert_id * expert_size,
                (size_t) slot * cache->slot_stride,
                copy_size);

        cache->expert_in_slot[slot] = expert_id;
        cache->slot_of[expert_id] = slot;
        cache->slot_speculative[slot] = speculative ? 1 : 0;
        cache->lru_tick[slot] = ++cache->now;
        return copy_size;
    };

    size_t copied_bytes = 0;
    for (int32_t expert_id : misses) {
        int32_t slot = find_free_slot();
        if (slot == -1) {
            slot = find_victim_slot(true);
        }

        if (slot == -1) {
            cache->bypasses++;
            return fail("no_evictable_slot");
        }

        const size_t copy_size = copy_expert_to_slot(expert_id, slot, false, false);
        copied_bytes += copy_size;
        cache->bytes_copied += copy_size;
    }

    for (int32_t expert_id : needed) {
        const int32_t slot = cache->slot_of[expert_id];
        GGML_ASSERT(slot >= 0);
        cache->lru_tick[slot] = ++cache->now;
    }

    cache->remapped_ids = ids;
    for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; i1++) {
        for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; i0++) {
            const int64_t idx = i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t);
            const int32_t expert_id = ids[idx];
            const int32_t slot = cache->slot_of[expert_id];
            GGML_ASSERT(slot >= 0);
            cache->remapped_ids[idx] = slot;
        }
    }

    ggml_backend_tensor_set_async(split_backend, &cache->ids_tensor, cache->remapped_ids.data(), 0, cache->ids_nbytes);

    size_t prefetched_bytes = 0;
    size_t prefetch_count = 0;
    if (prefetch_policy == GGML_BACKEND_SCHED_MOE_PREFETCH_SETMARKOV && needed.size() <= 64) {
        if (!cache->previous_experts.empty() && cache->previous_experts.size() <= 64) {
            ggml_backend_sched_moe_transition * transition = nullptr;
            for (ggml_backend_sched_moe_transition & candidate : cache->transitions) {
                if (candidate.from == cache->previous_experts) {
                    transition = &candidate;
                    break;
                }
            }
            if (transition == nullptr) {
                cache->transitions.push_back({ cache->previous_experts, std::vector<uint32_t>((size_t) n_expert, 0) });
                transition = &cache->transitions.back();
            }

            for (int32_t expert_id : needed) {
                uint32_t & count = transition->counts[expert_id];
                if (count < std::numeric_limits<uint32_t>::max()) {
                    count++;
                }
            }
        }

        ggml_backend_sched_moe_transition * prediction = nullptr;
        for (ggml_backend_sched_moe_transition & candidate : cache->transitions) {
            if (candidate.from == needed) {
                prediction = &candidate;
                break;
            }
        }

        if (prediction != nullptr) {
            std::vector<std::pair<int32_t, uint32_t>> candidates;
            candidates.reserve((size_t) n_expert);
            for (int64_t expert_id = 0; expert_id < n_expert; ++expert_id) {
                const uint32_t count = prediction->counts[expert_id];
                if (count > 0) {
                    candidates.push_back({ (int32_t) expert_id, count });
                }
            }
            std::sort(candidates.begin(), candidates.end(),
                    [](const std::pair<int32_t, uint32_t> & a, const std::pair<int32_t, uint32_t> & b) {
                if (a.second != b.second) {
                    return a.second > b.second;
                }
                return a.first < b.first;
            });

            if (prefetch_limit > 0) {
                const size_t limit = std::min<size_t>((size_t) std::min(prefetch_limit, cache->n_slots), candidates.size());
                for (size_t i = 0; i < limit; ++i) {
                    const int32_t expert_id = candidates[i].first;
                    int32_t slot = cache->slot_of[expert_id];
                    if (slot >= 0) {
                        cache->lru_tick[slot] = ++cache->now;
                        continue;
                    }

                    slot = find_free_slot();
                    if (slot == -1) {
                        slot = find_victim_slot(true);
                    }
                    if (slot == -1) {
                        break;
                    }

                    const size_t copy_size = copy_expert_to_slot(expert_id, slot, true, true);
                    prefetched_bytes += copy_size;
                    cache->bytes_prefetched += copy_size;
                    cache->prefetches++;
                    prefetch_count++;
                }
            }
        }

        cache->previous_experts = needed;
    }

    restores.push_back({ node, node->src[0], node->src[2] });
    node->src[0] = &cache->weights_tensor;
    node->src[2] = &cache->ids_tensor;

    if (moe_log) {
        GGML_LOG_INFO("%s: moe_cache tensor=%s backend=%s slots=%d expert_size=%zu cache_bytes=%zu used=%zu hits=%zu misses=%zu copied=%zu prefetches=%zu prefetched=%zu total_hits=%llu total_speculative_hits=%llu total_misses=%llu total_copied=%llu total_prefetches=%llu total_wrong_prefetches=%llu total_prefetch_evictions=%llu total_prefetched=%llu\n",
                __func__,
                ggml_backend_sched_tensor_name(input),
                ggml_backend_name(split_backend),
                cache->n_slots,
                cache->expert_size,
                cache->weights_size,
                needed.size(),
                needed.size() - misses.size(),
                misses.size(),
                copied_bytes,
                prefetch_count,
                prefetched_bytes,
                (unsigned long long) cache->hits,
                (unsigned long long) cache->speculative_hits,
                (unsigned long long) cache->misses,
                (unsigned long long) cache->bytes_copied,
                (unsigned long long) cache->prefetches,
                (unsigned long long) cache->wrong_prefetches,
                (unsigned long long) cache->prefetch_evictions,
                (unsigned long long) cache->bytes_prefetched);
    }

    return true;
}

static ggml_backend_buffer_t ggml_backend_sched_kimi_tensor_buffer(
        const ggml_tensor * tensor) {
    return tensor->view_src != nullptr ? tensor->view_src->buffer : tensor->buffer;
}

static void ggml_backend_sched_kimi_async_d2h_fail_locked(
        ggml_backend_sched_kimi_async_d2h_state * state,
        const char * reason) {
    state->failed = true;
    state->failure_reason = reason;
    state->state_cv.notify_all();
    state->work_cv.notify_all();
}

static void ggml_backend_sched_kimi_async_d2h_worker(
        ggml_backend_sched_kimi_async_d2h_state * state) {
    while (true) {
        ggml_backend_sched_kimi_async_d2h_task task;
        {
            std::unique_lock<std::mutex> lock(state->mutex);
            state->work_cv.wait(lock, [&]() {
                return state->stop || !state->tasks.empty();
            });
            if (state->stop && state->tasks.empty()) {
                break;
            }
            task = state->tasks.front();
            state->tasks.pop_front();
            state->active_tasks++;
            state->worker_wakeups++;

            if (task.edge >= state->edges.size() ||
                    task.slot < 0 || task.slot >= state->N_SLOTS) {
                state->ownership_failures++;
                ggml_backend_sched_kimi_async_d2h_fail_locked(
                    state, "worker-task-identity");
                state->active_tasks--;
                state->state_cv.notify_all();
                continue;
            }
            ggml_backend_sched_kimi_async_d2h_edge & edge =
                state->edges[task.edge];
            ggml_backend_sched_kimi_async_d2h_slot & slot =
                state->slots[task.slot];
            if (!slot.busy || slot.generation != task.generation ||
                    edge.generation != task.generation ||
                    edge.slot != task.slot || edge.src != task.src ||
                    edge.bytes != task.bytes ||
                    edge.state != GGML_KIMI_ASYNC_D2H_SOURCE_PUBLISHED) {
                state->stale_generation_failures++;
                ggml_backend_sched_kimi_async_d2h_fail_locked(
                    state, "worker-stale-generation");
                state->active_tasks--;
                state->state_cv.notify_all();
                continue;
            }
            edge.state = GGML_KIMI_ASYNC_D2H_TRANSFER_ENQUEUED;
            state->transfer_enqueued++;
        }

        ggml_backend_sched_kimi_async_d2h_slot & slot =
            state->slots[task.slot];
        ggml_backend_event_wait(state->backend, slot.producer_ready);
        state->producer_waits++;
        ggml_backend_tensor_get_async(
            state->backend,
            task.src,
            ggml_backend_buffer_get_base(slot.staging),
            0,
            task.bytes);
        state->d2h_calls++;
        state->d2h_bytes += task.bytes;
        ggml_backend_event_record(slot.d2h_done, state->backend);
        state->done_records++;
        ggml_backend_event_synchronize(slot.d2h_done);
        state->done_host_waits++;

        {
            std::lock_guard<std::mutex> lock(state->mutex);
            ggml_backend_sched_kimi_async_d2h_edge & edge =
                state->edges[task.edge];
            if (!slot.busy || slot.generation != task.generation ||
                    edge.generation != task.generation ||
                    edge.slot != task.slot ||
                    edge.state != GGML_KIMI_ASYNC_D2H_TRANSFER_ENQUEUED) {
                state->stale_generation_failures++;
                ggml_backend_sched_kimi_async_d2h_fail_locked(
                    state, "worker-completion-generation");
            } else {
                edge.state = GGML_KIMI_ASYNC_D2H_READY_PUBLISHED;
                state->ready_published++;
            }
            state->active_tasks--;
            state->state_cv.notify_all();
        }
    }
}

static void ggml_backend_sched_kimi_async_d2h_release_resources(
        ggml_backend_sched_kimi_async_d2h_state * state) {
    for (ggml_backend_sched_kimi_async_d2h_slot & slot : state->slots) {
        ggml_backend_event_free(slot.producer_ready);
        ggml_backend_event_free(slot.d2h_done);
        ggml_backend_buffer_free(slot.staging);
        slot.producer_ready = nullptr;
        slot.d2h_done = nullptr;
        slot.staging = nullptr;
        slot.busy = false;
    }
    ggml_backend_free(state->backend);
    state->backend = nullptr;
    state->device = nullptr;
    state->host_buft = nullptr;
    state->staging_capacity = 0;
}

static bool ggml_backend_sched_kimi_async_d2h_initialize(
        ggml_backend_sched_kimi_async_d2h_state * state,
        ggml_backend_dev_t device,
        size_t staging_capacity) {
    if (state->worker_started) {
        return state->device == device &&
            state->staging_capacity >= staging_capacity;
    }

    state->device = device;
    state->host_buft = ggml_backend_dev_host_buffer_type(device);
    state->backend = ggml_backend_dev_init(device, nullptr);
    if (state->host_buft == nullptr || state->backend == nullptr ||
            state->backend->iface.get_tensor_async == nullptr ||
            state->backend->iface.event_wait == nullptr ||
            state->backend->iface.event_record == nullptr) {
        state->init_failures++;
        ggml_backend_sched_kimi_async_d2h_release_resources(state);
        return false;
    }

    state->staging_capacity = staging_capacity;
    for (ggml_backend_sched_kimi_async_d2h_slot & slot : state->slots) {
        slot.staging = ggml_backend_buft_alloc_buffer(
            state->host_buft, state->staging_capacity);
        slot.producer_ready = ggml_backend_event_new(device);
        slot.d2h_done = ggml_backend_event_new(device);
        if (slot.staging == nullptr || slot.producer_ready == nullptr ||
                slot.d2h_done == nullptr ||
                ggml_backend_buffer_get_type(slot.staging) != state->host_buft ||
                !ggml_backend_buffer_is_host(slot.staging) ||
                ggml_backend_buffer_get_base(slot.staging) == nullptr) {
            state->init_failures++;
            ggml_backend_sched_kimi_async_d2h_release_resources(state);
            return false;
        }
    }

    try {
        state->worker = std::thread(
            ggml_backend_sched_kimi_async_d2h_worker, state);
        state->worker_started = true;
    } catch (...) {
        state->init_failures++;
        ggml_backend_sched_kimi_async_d2h_release_resources(state);
        return false;
    }

    GGML_LOG_INFO(
        "%s: async split-input D2H worker enabled device=%s slots=%d "
        "staging_capacity=%zu\n",
        __func__, ggml_backend_dev_name(device), state->N_SLOTS,
        state->staging_capacity);
    return true;
}

static bool ggml_backend_sched_kimi_async_d2h_build_plan(
        ggml_backend_sched_t sched,
        ggml_backend_sched_kimi_async_d2h_state * state,
        const char ** decline_reason,
        ggml_backend_dev_t * plan_device,
        size_t * staging_capacity) {
    if (sched->n_copies != 1) {
        *decline_reason = "scheduler-copies";
        return false;
    }
    if (sched->callback_eval != nullptr) {
        *decline_reason = "callback-eval";
        return false;
    }
    if (sched->n_splits != state->EXPECTED_SPLITS) {
        *decline_reason = "non-canonical-split-count";
        return false;
    }

    std::map<const ggml_tensor *, int> producer_by_tensor;
    for (int split_id = 0; split_id < sched->n_splits; ++split_id) {
        const ggml_backend_sched_split & split = sched->splits[split_id];
        for (int node_id = 0; node_id < split.graph.n_nodes; ++node_id) {
            ggml_tensor * node = split.graph.nodes[node_id];
            auto inserted = producer_by_tensor.emplace(node, split_id);
            if (!inserted.second && inserted.first->second != split_id) {
                *decline_reason = "duplicate-producer";
                return false;
            }
        }
    }

    std::vector<ggml_backend_sched_kimi_async_d2h_edge> edges;
    std::vector<int> edge_by_split_input(
        sched->n_splits * GGML_SCHED_MAX_SPLIT_INPUTS, -1);
    ggml_backend_dev_t device = nullptr;
    size_t max_bytes = 0;

    for (int split_id = 0; split_id < sched->n_splits; ++split_id) {
        ggml_backend_sched_split & split = sched->splits[split_id];
        ggml_backend_t dst_backend = sched->backends[split.backend_id];
        if (ggml_backend_dev_type(ggml_backend_get_device(dst_backend)) !=
                GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        for (int input_id = 0; input_id < split.n_inputs; ++input_id) {
            ggml_tensor * src = split.inputs[input_id];
            ggml_backend_t src_backend =
                ggml_backend_sched_get_tensor_backend(sched, src);
            ggml_backend_dev_t src_device =
                ggml_backend_get_device(src_backend);
            if (ggml_backend_dev_type(src_device) !=
                    GGML_BACKEND_DEVICE_TYPE_GPU) {
                continue;
            }

            ggml_tensor * dst = tensor_copy(
                src, split.backend_id, sched->cur_copy);
            ggml_backend_buffer_t src_buffer =
                ggml_backend_sched_kimi_tensor_buffer(src);
            ggml_backend_buffer_t dst_buffer =
                ggml_backend_sched_kimi_tensor_buffer(dst);
            auto producer = producer_by_tensor.find(src);
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                *decline_reason = "user-input";
                return false;
            }
            if (src_buffer == nullptr || dst_buffer == nullptr ||
                    src->data == nullptr || dst->data == nullptr) {
                *decline_reason = "unallocated-tensor";
                return false;
            }
            if (ggml_backend_buffer_get_usage(src_buffer) ==
                    GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                *decline_reason = "weight-input";
                return false;
            }
            if (producer == producer_by_tensor.end() ||
                    producer->second >= split_id) {
                *decline_reason = "producer-order";
                return false;
            }
            if (device != nullptr && device != src_device) {
                *decline_reason = "cross-device-direction";
                return false;
            }
            if (src_backend->iface.get_tensor_async == nullptr ||
                    src_backend->iface.event_record == nullptr ||
                    src_device->iface.event_new == nullptr ||
                    ggml_backend_buffer_get_type(src_buffer) !=
                        ggml_backend_dev_buffer_type(src_device)) {
                *decline_reason = "source-capability";
                return false;
            }
            if (!ggml_backend_buffer_is_host(dst_buffer)) {
                *decline_reason = "destination-not-host";
                return false;
            }
            if (!ggml_are_same_layout(src, dst) ||
                    ggml_nbytes(src) != ggml_nbytes(dst) ||
                    ggml_nbytes(src) == 0) {
                *decline_reason = "layout-or-size";
                return false;
            }

            device = src_device;
            ggml_backend_sched_kimi_async_d2h_edge edge;
            edge.src = src;
            edge.dst = dst;
            edge.src_backend = src_backend;
            edge.src_buffer = src_buffer;
            edge.dst_buffer = dst_buffer;
            edge.producer_split = producer->second;
            edge.consumer_split = split_id;
            edge.input_id = input_id;
            edge.bytes = ggml_nbytes(src);
            const int edge_id = (int) edges.size();
            edge_by_split_input[
                split_id * GGML_SCHED_MAX_SPLIT_INPUTS + input_id] = edge_id;
            edges.push_back(edge);
            max_bytes = std::max(max_bytes, edge.bytes);
        }
    }

    if ((int) edges.size() != state->EXPECTED_EDGES || device == nullptr ||
            max_bytes == 0) {
        *decline_reason = "non-canonical-edge-count";
        return false;
    }

    state->edges = std::move(edges);
    state->edge_by_split_input = std::move(edge_by_split_input);
    *plan_device = device;
    *staging_capacity = max_bytes;
    return true;
}

static bool ggml_backend_sched_kimi_async_d2h_begin_graph(
        ggml_backend_sched_t sched) {
    ggml_backend_sched_kimi_async_d2h_state * state = sched->kimi_async_d2h;
    if (state == nullptr) {
        return true;
    }
    state->active_graph = false;
    state->edges.clear();
    state->edge_by_split_input.clear();

    // Phase 2 slice 1 is deliberately limited to canonical single-token
    // decode. Prompt, reserve/probe and callback graphs retain the legacy path.
    if (g_kimi_split_profile_phase != 2 ||
            !sched->decode_phase) {
        return true;
    }

    state->graphs_started++;
    if (state->failed) {
        state->graphs_declined++;
        state->last_decline_reason = "state-poisoned";
        return true;
    }

    const char * decline_reason = "none";
    ggml_backend_dev_t device = nullptr;
    size_t staging_capacity = 0;
    if (!ggml_backend_sched_kimi_async_d2h_build_plan(
                sched, state, &decline_reason, &device, &staging_capacity)) {
        state->graphs_declined++;
        state->last_decline_reason = decline_reason;
        if (!state->decline_logged) {
            GGML_LOG_WARN(
                "%s: async split-input D2H declined reason=%s splits=%d\n",
                __func__, decline_reason, sched->n_splits);
            state->decline_logged = true;
        }
        state->edges.clear();
        state->edge_by_split_input.clear();
        return true;
    }
    if (!ggml_backend_sched_kimi_async_d2h_initialize(
                state, device, staging_capacity)) {
        state->graphs_declined++;
        state->last_decline_reason = "transport-init";
        state->edges.clear();
        state->edge_by_split_input.clear();
        return true;
    }

    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!state->tasks.empty() || state->active_tasks != 0) {
            state->ownership_failures++;
            ggml_backend_sched_kimi_async_d2h_fail_locked(
                state, "graph-begin-pending-work");
            return false;
        }
        for (const ggml_backend_sched_kimi_async_d2h_slot & slot :
                state->slots) {
            if (slot.busy) {
                state->ownership_failures++;
                ggml_backend_sched_kimi_async_d2h_fail_locked(
                    state, "graph-begin-busy-slot");
                return false;
            }
        }
    }
    state->graph_epoch++;
    state->next_submit = 0;
    state->producer_frontier = -1;
    state->active_graph = true;
    state->graphs_activated++;
    state->descriptors_planned += state->edges.size();
    return true;
}

static int ggml_backend_sched_kimi_async_d2h_free_slot(
        ggml_backend_sched_kimi_async_d2h_state * state) {
    std::lock_guard<std::mutex> lock(state->mutex);
    for (int slot_id = 0; slot_id < state->N_SLOTS; ++slot_id) {
        if (!state->slots[slot_id].busy) {
            return slot_id;
        }
    }
    return -1;
}

static bool ggml_backend_sched_kimi_async_d2h_submit(
        ggml_backend_sched_kimi_async_d2h_state * state,
        size_t edge_id,
        int slot_id) {
    uint64_t generation = 0;
    ggml_backend_t src_backend = nullptr;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->failed || edge_id >= state->edges.size() ||
                slot_id < 0 || slot_id >= state->N_SLOTS ||
                state->slots[slot_id].busy) {
            state->ownership_failures++;
            ggml_backend_sched_kimi_async_d2h_fail_locked(
                state, "submit-ownership");
            return false;
        }
        ggml_backend_sched_kimi_async_d2h_edge & edge = state->edges[edge_id];
        if (edge.state != GGML_KIMI_ASYNC_D2H_PLANNED) {
            state->reorder_failures++;
            ggml_backend_sched_kimi_async_d2h_fail_locked(
                state, "submit-reorder");
            return false;
        }
        generation = ++state->next_generation;
        edge.generation = generation;
        edge.slot = slot_id;
        edge.state = GGML_KIMI_ASYNC_D2H_RESERVED;
        ggml_backend_sched_kimi_async_d2h_slot & slot = state->slots[slot_id];
        slot.busy = true;
        slot.generation = generation;
        src_backend = edge.src_backend;
        state->descriptors_reserved++;
    }

    ggml_backend_event_record(
        state->slots[slot_id].producer_ready, src_backend);

    {
        std::lock_guard<std::mutex> lock(state->mutex);
        ggml_backend_sched_kimi_async_d2h_edge & edge = state->edges[edge_id];
        ggml_backend_sched_kimi_async_d2h_slot & slot = state->slots[slot_id];
        if (state->failed || edge.generation != generation ||
                edge.state != GGML_KIMI_ASYNC_D2H_RESERVED ||
                !slot.busy || slot.generation != generation) {
            state->stale_generation_failures++;
            ggml_backend_sched_kimi_async_d2h_fail_locked(
                state, "submit-publish-generation");
            return false;
        }
        edge.state = GGML_KIMI_ASYNC_D2H_SOURCE_PUBLISHED;
        state->producer_records++;
        state->descriptors_submitted++;
        state->tasks.push_back({
            edge_id, slot_id, generation, edge.src, edge.bytes,
        });
        state->queue_max = std::max<uint64_t>(
            state->queue_max, state->tasks.size());
        state->work_cv.notify_one();
    }
    return true;
}

static bool ggml_backend_sched_kimi_async_d2h_pump(
        ggml_backend_sched_kimi_async_d2h_state * state) {
    if (!state->active_graph) {
        return true;
    }
    while (state->next_submit < state->edges.size()) {
        const ggml_backend_sched_kimi_async_d2h_edge & edge =
            state->edges[state->next_submit];
        if (edge.producer_split > state->producer_frontier) {
            return true;
        }
        const int slot_id =
            ggml_backend_sched_kimi_async_d2h_free_slot(state);
        if (slot_id < 0) {
            state->deferred_submissions++;
            return true;
        }
        if (!ggml_backend_sched_kimi_async_d2h_submit(
                    state, state->next_submit, slot_id)) {
            return false;
        }
        state->next_submit++;
    }
    return true;
}

static bool ggml_backend_sched_kimi_async_d2h_producer_submitted(
        ggml_backend_sched_t sched,
        int split_id) {
    ggml_backend_sched_kimi_async_d2h_state * state = sched->kimi_async_d2h;
    if (state == nullptr || !state->active_graph) {
        return true;
    }
    if (split_id != state->producer_frontier + 1) {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->reorder_failures++;
        ggml_backend_sched_kimi_async_d2h_fail_locked(
            state, "producer-frontier");
        return false;
    }
    state->producer_frontier = split_id;
    return ggml_backend_sched_kimi_async_d2h_pump(state);
}

static int ggml_backend_sched_kimi_async_d2h_edge_id(
        const ggml_backend_sched_kimi_async_d2h_state * state,
        int split_id,
        int input_id) {
    if (!state->active_graph || split_id < 0 || input_id < 0 ||
            split_id >= ggml_backend_sched_kimi_async_d2h_state::EXPECTED_SPLITS ||
            input_id >= GGML_SCHED_MAX_SPLIT_INPUTS) {
        return -1;
    }
    const size_t index =
        split_id * GGML_SCHED_MAX_SPLIT_INPUTS + input_id;
    return index < state->edge_by_split_input.size() ?
        state->edge_by_split_input[index] : -1;
}

static bool ggml_backend_sched_kimi_async_d2h_prepare_input(
        ggml_backend_sched_t sched,
        int split_id,
        int input_id,
        ggml_tensor * src,
        ggml_tensor * dst) {
    ggml_backend_sched_kimi_async_d2h_state * state = sched->kimi_async_d2h;
    const int edge_id = state == nullptr ? -1 :
        ggml_backend_sched_kimi_async_d2h_edge_id(state, split_id, input_id);
    if (edge_id < 0) {
        return false;
    }

    while (true) {
        enum ggml_backend_sched_kimi_async_d2h_edge_state edge_state;
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (state->failed) {
                return false;
            }
            edge_state = state->edges[edge_id].state;
        }
        if (edge_state != GGML_KIMI_ASYNC_D2H_PLANNED) {
            break;
        }
        if (!ggml_backend_sched_kimi_async_d2h_pump(state)) {
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (state->edges[edge_id].state == GGML_KIMI_ASYNC_D2H_PLANNED) {
                state->ownership_failures++;
                ggml_backend_sched_kimi_async_d2h_fail_locked(
                    state, "consumer-without-slot");
                return false;
            }
        }
    }

    int slot_id = -1;
    uint64_t generation = 0;
    bool blocked = false;
    const int64_t wait_start_us = ggml_time_us();
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        blocked = state->edges[edge_id].state !=
            GGML_KIMI_ASYNC_D2H_READY_PUBLISHED;
        state->state_cv.wait(lock, [&]() {
            return state->failed ||
                state->edges[edge_id].state ==
                    GGML_KIMI_ASYNC_D2H_READY_PUBLISHED;
        });
        if (state->failed) {
            return false;
        }
        ggml_backend_sched_kimi_async_d2h_edge & edge = state->edges[edge_id];
        if (edge.src != src || edge.dst != dst || edge.bytes != ggml_nbytes(src) ||
                edge.bytes != ggml_nbytes(dst) ||
                edge.src_buffer != ggml_backend_sched_kimi_tensor_buffer(src) ||
                edge.dst_buffer != ggml_backend_sched_kimi_tensor_buffer(dst)) {
            state->identity_failures++;
            ggml_backend_sched_kimi_async_d2h_fail_locked(
                state, "consumer-identity");
            return false;
        }
        if (!ggml_are_same_layout(src, dst)) {
            state->layout_failures++;
            ggml_backend_sched_kimi_async_d2h_fail_locked(
                state, "consumer-layout");
            return false;
        }
        slot_id = edge.slot;
        generation = edge.generation;
        if (slot_id < 0 || slot_id >= state->N_SLOTS ||
                !state->slots[slot_id].busy ||
                state->slots[slot_id].generation != generation) {
            state->ownership_failures++;
            ggml_backend_sched_kimi_async_d2h_fail_locked(
                state, "consumer-slot-ownership");
            return false;
        }
    }
    const uint64_t wait_us = (uint64_t) (ggml_time_us() - wait_start_us);

    memcpy(
        dst->data,
        ggml_backend_buffer_get_base(state->slots[slot_id].staging),
        ggml_nbytes(dst));

    {
        std::lock_guard<std::mutex> lock(state->mutex);
        ggml_backend_sched_kimi_async_d2h_edge & edge = state->edges[edge_id];
        ggml_backend_sched_kimi_async_d2h_slot & slot = state->slots[slot_id];
        if (edge.state != GGML_KIMI_ASYNC_D2H_READY_PUBLISHED ||
                edge.generation != generation || !slot.busy ||
                slot.generation != generation) {
            state->stale_generation_failures++;
            ggml_backend_sched_kimi_async_d2h_fail_locked(
                state, "consumer-copy-generation");
            return false;
        }
        edge.state = GGML_KIMI_ASYNC_D2H_HOST_COPIED;
        edge.slot = -1;
        slot.busy = false;
        state->host_publish_calls++;
        state->host_publish_bytes += edge.bytes;
        if (blocked) {
            state->scheduler_ready_waits++;
            state->scheduler_ready_wait_us += wait_us;
            state->scheduler_ready_wait_max_us = std::max(
                state->scheduler_ready_wait_max_us, wait_us);
        }
        state->state_cv.notify_all();
    }
    return ggml_backend_sched_kimi_async_d2h_pump(state);
}

static bool ggml_backend_sched_kimi_async_d2h_consumer_begin(
        ggml_backend_sched_t sched,
        int split_id) {
    ggml_backend_sched_kimi_async_d2h_state * state = sched->kimi_async_d2h;
    if (state == nullptr || !state->active_graph) {
        return true;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    for (ggml_backend_sched_kimi_async_d2h_edge & edge : state->edges) {
        if (edge.consumer_split != split_id) {
            continue;
        }
        if (edge.state != GGML_KIMI_ASYNC_D2H_HOST_COPIED) {
            state->reorder_failures++;
            ggml_backend_sched_kimi_async_d2h_fail_locked(
                state, "consumer-launch-before-copy");
            return false;
        }
        edge.state = GGML_KIMI_ASYNC_D2H_CONSUMER_QUEUED;
        state->consumer_queued++;
    }
    return true;
}

static bool ggml_backend_sched_kimi_async_d2h_consumer_end(
        ggml_backend_sched_t sched,
        int split_id) {
    ggml_backend_sched_kimi_async_d2h_state * state = sched->kimi_async_d2h;
    if (state == nullptr || !state->active_graph) {
        return true;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    for (ggml_backend_sched_kimi_async_d2h_edge & edge : state->edges) {
        if (edge.consumer_split != split_id) {
            continue;
        }
        if (edge.state != GGML_KIMI_ASYNC_D2H_CONSUMER_QUEUED) {
            state->reorder_failures++;
            ggml_backend_sched_kimi_async_d2h_fail_locked(
                state, "consumer-retire-reorder");
            return false;
        }
        edge.state = GGML_KIMI_ASYNC_D2H_RETIRED;
        state->descriptors_retired++;
    }
    return true;
}

static bool ggml_backend_sched_kimi_async_d2h_finish_graph(
        ggml_backend_sched_t sched) {
    ggml_backend_sched_kimi_async_d2h_state * state = sched->kimi_async_d2h;
    if (state == nullptr || !state->active_graph) {
        return true;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    bool closed = !state->failed &&
        state->next_submit == state->edges.size() &&
        state->tasks.empty() && state->active_tasks == 0;
    for (const ggml_backend_sched_kimi_async_d2h_edge & edge : state->edges) {
        closed = closed && edge.state == GGML_KIMI_ASYNC_D2H_RETIRED;
    }
    for (const ggml_backend_sched_kimi_async_d2h_slot & slot : state->slots) {
        closed = closed && !slot.busy;
    }
    if (!closed) {
        state->ownership_failures++;
        ggml_backend_sched_kimi_async_d2h_fail_locked(
            state, "graph-ledger-open");
        return false;
    }
    state->graphs_completed++;
    state->active_graph = false;
    state->edges.clear();
    state->edge_by_split_input.clear();
    return true;
}

static bool ggml_backend_sched_kimi_async_d2h_drain(
        ggml_backend_sched_kimi_async_d2h_state * state,
        bool poison_active) {
    if (state == nullptr || !state->worker_started) {
        return state == nullptr || !state->active_graph;
    }
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        if (poison_active && state->active_graph) {
            state->graphs_poisoned++;
            ggml_backend_sched_kimi_async_d2h_fail_locked(
                state, "graph-aborted");
        }
        state->state_cv.wait(lock, [&]() {
            return state->tasks.empty() && state->active_tasks == 0;
        });
    }
    ggml_backend_synchronize(state->backend);
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->active_graph) {
            for (ggml_backend_sched_kimi_async_d2h_edge & edge : state->edges) {
                if (edge.state != GGML_KIMI_ASYNC_D2H_RETIRED) {
                    edge.state = GGML_KIMI_ASYNC_D2H_POISONED;
                }
            }
            for (ggml_backend_sched_kimi_async_d2h_slot & slot : state->slots) {
                slot.busy = false;
            }
            state->active_graph = false;
            state->edges.clear();
            state->edge_by_split_input.clear();
        }
    }
    return !state->failed;
}

static void ggml_backend_sched_kimi_async_d2h_shutdown(
        ggml_backend_sched_kimi_async_d2h_state * state) {
    if (state == nullptr) {
        return;
    }
    (void) ggml_backend_sched_kimi_async_d2h_drain(state, true);
    if (state->worker_started) {
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->stop = true;
            state->work_cv.notify_all();
        }
        state->worker.join();
        state->worker_started = false;
    }

    int slots_free = 0;
    for (const ggml_backend_sched_kimi_async_d2h_slot & slot : state->slots) {
        slots_free += !slot.busy;
    }
    const bool guards_closed = !state->failed &&
        state->graphs_activated == state->graphs_completed &&
        state->descriptors_planned == state->descriptors_reserved &&
        state->descriptors_reserved == state->descriptors_submitted &&
        state->descriptors_submitted == state->transfer_enqueued &&
        state->transfer_enqueued == state->ready_published &&
        state->ready_published == state->consumer_queued &&
        state->consumer_queued == state->descriptors_retired &&
        state->d2h_calls == state->host_publish_calls &&
        state->d2h_bytes == state->host_publish_bytes &&
        state->producer_records == state->producer_waits &&
        state->done_records == state->done_host_waits &&
        state->tasks.empty() && state->active_tasks == 0 &&
        slots_free == state->N_SLOTS;
    GGML_LOG_INFO(
        "%s: async_split_input_d2h default_off=0 graphs_started=%llu "
        "activated=%llu declined=%llu completed=%llu poisoned=%llu "
        "planned=%llu reserved=%llu submitted=%llu transfer_enqueued=%llu "
        "ready_published=%llu consumer_queued=%llu retired=%llu "
        "producer_records=%llu producer_waits=%llu done_records=%llu "
        "done_host_waits=%llu d2h_calls=%llu d2h_bytes=%llu "
        "host_publish_calls=%llu host_publish_bytes=%llu ready_waits=%llu "
        "ready_wait_us=%llu ready_wait_max_us=%llu deferred=%llu "
        "worker_wakeups=%llu queue_max=%llu slots_free=%d "
        "stale_failures=%llu identity_failures=%llu layout_failures=%llu "
        "ownership_failures=%llu reorder_failures=%llu init_failures=%llu "
        "failed=%d guards_closed=%d failure_reason=%s decline_reason=%s\n",
        __func__,
        (unsigned long long) state->graphs_started,
        (unsigned long long) state->graphs_activated,
        (unsigned long long) state->graphs_declined,
        (unsigned long long) state->graphs_completed,
        (unsigned long long) state->graphs_poisoned,
        (unsigned long long) state->descriptors_planned,
        (unsigned long long) state->descriptors_reserved,
        (unsigned long long) state->descriptors_submitted,
        (unsigned long long) state->transfer_enqueued,
        (unsigned long long) state->ready_published,
        (unsigned long long) state->consumer_queued,
        (unsigned long long) state->descriptors_retired,
        (unsigned long long) state->producer_records,
        (unsigned long long) state->producer_waits,
        (unsigned long long) state->done_records,
        (unsigned long long) state->done_host_waits,
        (unsigned long long) state->d2h_calls,
        (unsigned long long) state->d2h_bytes,
        (unsigned long long) state->host_publish_calls,
        (unsigned long long) state->host_publish_bytes,
        (unsigned long long) state->scheduler_ready_waits,
        (unsigned long long) state->scheduler_ready_wait_us,
        (unsigned long long) state->scheduler_ready_wait_max_us,
        (unsigned long long) state->deferred_submissions,
        (unsigned long long) state->worker_wakeups,
        (unsigned long long) state->queue_max,
        slots_free,
        (unsigned long long) state->stale_generation_failures,
        (unsigned long long) state->identity_failures,
        (unsigned long long) state->layout_failures,
        (unsigned long long) state->ownership_failures,
        (unsigned long long) state->reorder_failures,
        (unsigned long long) state->init_failures,
        state->failed ? 1 : 0,
        guards_closed ? 1 : 0,
        state->failure_reason,
        state->last_decline_reason);

    ggml_backend_sched_kimi_async_d2h_release_resources(state);
}

static void ggml_backend_sched_kimi_async_h2d_fail_locked(
        ggml_backend_sched_kimi_async_h2d_state * state,
        const char * reason) {
    state->failed = true;
    state->failure_reason = reason;
    state->state_cv.notify_all();
    state->work_cv.notify_all();
}

static void ggml_backend_sched_kimi_async_h2d_worker(
        ggml_backend_sched_kimi_async_h2d_state * state) {
    while (true) {
        ggml_backend_sched_kimi_async_h2d_task task;
        {
            std::unique_lock<std::mutex> lock(state->mutex);
            state->work_cv.wait(lock, [&]() {
                return state->stop || !state->tasks.empty();
            });
            if (state->stop && state->tasks.empty()) {
                break;
            }
            task = state->tasks.front();
            state->tasks.pop_front();
            state->active_tasks++;
            state->worker_wakeups++;
            if (task.edge >= state->edges.size() || task.slot < 0 ||
                    task.slot >= state->N_SLOTS) {
                state->ownership_failures++;
                ggml_backend_sched_kimi_async_h2d_fail_locked(
                    state, "worker-task-identity");
                state->active_tasks--;
                state->state_cv.notify_all();
                continue;
            }
            ggml_backend_sched_kimi_async_h2d_edge & edge =
                state->edges[task.edge];
            ggml_backend_sched_kimi_async_h2d_slot & slot =
                state->slots[task.slot];
            const enum ggml_backend_sched_kimi_async_h2d_edge_state expected =
                task.transfer_phase ?
                    GGML_KIMI_ASYNC_H2D_RELEASE_PUBLISHED :
                    GGML_KIMI_ASYNC_H2D_RESERVED;
            if (!slot.busy || slot.generation != task.generation ||
                    edge.generation != task.generation ||
                    edge.slot != task.slot || edge.src != task.src ||
                    edge.dst != task.dst || edge.src_backend != task.src_backend ||
                    edge.bytes != task.bytes ||
                    edge.state != expected) {
                state->stale_generation_failures++;
                ggml_backend_sched_kimi_async_h2d_fail_locked(
                    state, "worker-stale-generation");
                state->active_tasks--;
                state->state_cv.notify_all();
                continue;
            }
        }

        ggml_backend_sched_kimi_async_h2d_slot & slot =
            state->slots[task.slot];
        if (!task.transfer_phase) {
            if (task.reuse_slot) {
                ggml_backend_event_synchronize(slot.input_ready);
                state->reuse_host_waits++;
            }
            ggml_backend_synchronize(task.src_backend);
            state->source_syncs++;
            memcpy(
                ggml_backend_buffer_get_base(slot.staging),
                task.src->data,
                task.bytes);
            state->snapshot_calls++;
            state->snapshot_bytes += task.bytes;

            std::lock_guard<std::mutex> lock(state->mutex);
            ggml_backend_sched_kimi_async_h2d_edge & edge =
                state->edges[task.edge];
            if (!slot.busy || slot.generation != task.generation ||
                    edge.generation != task.generation ||
                    edge.slot != task.slot ||
                    edge.state != GGML_KIMI_ASYNC_H2D_RESERVED) {
                state->stale_generation_failures++;
                ggml_backend_sched_kimi_async_h2d_fail_locked(
                    state, "worker-snapshot-generation");
            } else {
                edge.state = GGML_KIMI_ASYNC_H2D_SOURCE_SNAPSHOTTED;
            }
            state->active_tasks--;
            state->state_cv.notify_all();
            continue;
        }

        // The compute backend records this release only at the original
        // consumer frontier.  Waiting here avoids overwriting an allocation
        // range that an intervening split may still alias.
        ggml_backend_event_wait(state->backend, slot.destination_release);
        state->release_waits++;
        ggml_backend_tensor_set_async(
            state->backend,
            task.dst,
            ggml_backend_buffer_get_base(slot.staging),
            0,
            task.bytes);
        state->h2d_calls++;
        state->h2d_bytes += task.bytes;
        ggml_backend_event_record(slot.input_ready, state->backend);
        state->ready_records++;

        {
            std::lock_guard<std::mutex> lock(state->mutex);
            ggml_backend_sched_kimi_async_h2d_edge & edge =
                state->edges[task.edge];
            if (!slot.busy || slot.generation != task.generation ||
                    edge.generation != task.generation ||
                    edge.slot != task.slot ||
                    edge.state != GGML_KIMI_ASYNC_H2D_RELEASE_PUBLISHED) {
                state->stale_generation_failures++;
                ggml_backend_sched_kimi_async_h2d_fail_locked(
                    state, "worker-ready-generation");
            } else {
                edge.state = GGML_KIMI_ASYNC_H2D_READY_PUBLISHED;
                state->transfer_enqueued++;
                state->ready_published++;
            }
            state->active_tasks--;
            state->state_cv.notify_all();
        }
    }
}

static void ggml_backend_sched_kimi_async_h2d_release_resources(
        ggml_backend_sched_kimi_async_h2d_state * state) {
    for (ggml_backend_sched_kimi_async_h2d_slot & slot : state->slots) {
        ggml_backend_event_free(slot.destination_release);
        ggml_backend_event_free(slot.input_ready);
        ggml_backend_buffer_free(slot.staging);
        slot.destination_release = nullptr;
        slot.input_ready = nullptr;
        slot.staging = nullptr;
        slot.busy = false;
        slot.has_previous = false;
    }
    ggml_backend_free(state->backend);
    state->backend = nullptr;
    state->device = nullptr;
    state->host_buft = nullptr;
    state->staging_capacity = 0;
}

static bool ggml_backend_sched_kimi_async_h2d_initialize(
        ggml_backend_sched_kimi_async_h2d_state * state,
        ggml_backend_dev_t device,
        size_t staging_capacity) {
    if (state->worker_started) {
        return state->device == device &&
            state->staging_capacity >= staging_capacity;
    }
    state->device = device;
    state->host_buft = ggml_backend_dev_host_buffer_type(device);
    state->backend = ggml_backend_dev_init(device, nullptr);
    if (state->host_buft == nullptr || state->backend == nullptr ||
            state->backend->iface.set_tensor_async == nullptr ||
            state->backend->iface.event_wait == nullptr ||
            state->backend->iface.event_record == nullptr) {
        state->init_failures++;
        ggml_backend_sched_kimi_async_h2d_release_resources(state);
        return false;
    }
    state->staging_capacity = staging_capacity;
    for (ggml_backend_sched_kimi_async_h2d_slot & slot : state->slots) {
        slot.staging = ggml_backend_buft_alloc_buffer(
            state->host_buft, state->staging_capacity);
        slot.destination_release = ggml_backend_event_new(device);
        slot.input_ready = ggml_backend_event_new(device);
        if (slot.staging == nullptr || slot.destination_release == nullptr ||
                slot.input_ready == nullptr ||
                ggml_backend_buffer_get_type(slot.staging) != state->host_buft ||
                !ggml_backend_buffer_is_host(slot.staging) ||
                ggml_backend_buffer_get_base(slot.staging) == nullptr) {
            state->init_failures++;
            ggml_backend_sched_kimi_async_h2d_release_resources(state);
            return false;
        }
    }
    try {
        state->worker = std::thread(
            ggml_backend_sched_kimi_async_h2d_worker, state);
        state->worker_started = true;
    } catch (...) {
        state->init_failures++;
        ggml_backend_sched_kimi_async_h2d_release_resources(state);
        return false;
    }
    GGML_LOG_INFO(
        "%s: async split-input H2D worker enabled device=%s slots=%d "
        "staging_capacity=%zu\n",
        __func__, ggml_backend_dev_name(device), state->N_SLOTS,
        state->staging_capacity);
    return true;
}

static bool ggml_backend_sched_kimi_async_h2d_build_plan(
        ggml_backend_sched_t sched,
        ggml_backend_sched_kimi_async_h2d_state * state,
        const char ** decline_reason,
        ggml_backend_dev_t * plan_device,
        size_t * staging_capacity) {
    if (sched->n_copies != 1) {
        *decline_reason = "scheduler-copies";
        return false;
    }
    if (sched->callback_eval != nullptr) {
        *decline_reason = "callback-eval";
        return false;
    }
    if (sched->n_splits != state->EXPECTED_SPLITS) {
        *decline_reason = "non-canonical-split-count";
        return false;
    }

    std::map<const ggml_tensor *, int> producer_by_tensor;
    for (int split_id = 0; split_id < sched->n_splits; ++split_id) {
        const ggml_backend_sched_split & split = sched->splits[split_id];
        for (int node_id = 0; node_id < split.graph.n_nodes; ++node_id) {
            ggml_tensor * node = split.graph.nodes[node_id];
            auto inserted = producer_by_tensor.emplace(node, split_id);
            if (!inserted.second && inserted.first->second != split_id) {
                *decline_reason = "duplicate-producer";
                return false;
            }
        }
    }

    std::vector<ggml_backend_sched_kimi_async_h2d_edge> edges;
    ggml_backend_dev_t device = nullptr;
    size_t max_bytes = 0;
    for (int split_id = 0; split_id < sched->n_splits; ++split_id) {
        ggml_backend_sched_split & split = sched->splits[split_id];
        ggml_backend_t dst_backend = sched->backends[split.backend_id];
        ggml_backend_dev_t dst_device = ggml_backend_get_device(dst_backend);
        if (ggml_backend_dev_type(dst_device) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        for (int input_id = 0; input_id < split.n_inputs; ++input_id) {
            ggml_tensor * src = split.inputs[input_id];
            ggml_backend_t src_backend =
                ggml_backend_sched_get_tensor_backend(sched, src);
            if (ggml_backend_dev_type(ggml_backend_get_device(src_backend)) !=
                    GGML_BACKEND_DEVICE_TYPE_CPU) {
                continue;
            }
            ggml_backend_buffer_t src_buffer =
                ggml_backend_sched_kimi_tensor_buffer(src);
            if (src_buffer == nullptr) {
                continue;
            }
            ggml_tensor * dst = tensor_copy(
                src, split.backend_id, sched->cur_copy);
            ggml_backend_buffer_t dst_buffer =
                ggml_backend_sched_kimi_tensor_buffer(dst);
            if (ggml_backend_buffer_get_usage(src_buffer) ==
                    GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(src_buffer)) {
                bool partial_moe_copy = false;
                for (int node_id = 0; node_id < split.graph.n_nodes;
                        ++node_id) {
                    const ggml_tensor * node = split.graph.nodes[node_id];
                    if (node->op == GGML_OP_MUL_MAT_ID &&
                            node->src[0] == dst) {
                        partial_moe_copy = true;
                        break;
                    }
                }
                if (partial_moe_copy) {
                    continue;
                }
            }
            auto producer = producer_by_tensor.find(src);
            const ggml_tensor * view_root =
                ggml_backend_sched_kimi_view_root(src);
            auto root_producer = producer_by_tensor.find(view_root);
            if (root_producer != producer_by_tensor.end() &&
                    (producer == producer_by_tensor.end() ||
                     root_producer->second < producer->second)) {
                producer = root_producer;
            }
            if (producer != producer_by_tensor.end() &&
                    producer->second >= split_id &&
                    ggml_is_view_op(src->op)) {
                // View/reshape nodes may be listed in the consumer split even
                // though their host storage is an already materialized leaf.
                // They perform no numerical work, so use the conservative
                // consumer-1 snapshot frontier.
                producer = producer_by_tensor.end();
            }
            if (producer != producer_by_tensor.end() &&
                    producer->second >= split_id) {
                *decline_reason = "producer-order";
                return false;
            }
            if (src->data == nullptr || dst->data == nullptr ||
                    !ggml_backend_buffer_is_host(src_buffer) ||
                    dst_buffer == nullptr ||
                    ggml_backend_buffer_get_type(dst_buffer) !=
                        ggml_backend_dev_buffer_type(dst_device) ||
                    dst_backend->iface.event_record == nullptr ||
                    dst_device->iface.event_new == nullptr) {
                *decline_reason = "edge-capability";
                return false;
            }
            if (!ggml_are_same_layout(src, dst) ||
                    ggml_nbytes(src) != ggml_nbytes(dst) ||
                    ggml_nbytes(src) == 0) {
                *decline_reason = "layout-or-size";
                return false;
            }
            if (device != nullptr && device != dst_device) {
                *decline_reason = "cross-device-direction";
                return false;
            }
            device = dst_device;
            ggml_backend_sched_kimi_async_h2d_edge edge;
            edge.src = src;
            edge.dst = dst;
            edge.src_backend = src_backend;
            edge.dst_backend = dst_backend;
            edge.src_buffer = src_buffer;
            edge.dst_buffer = dst_buffer;
            // Publish every snapshot only after the split immediately before
            // its consumer.  This keeps a two-slot staging ring bounded even
            // when a mapped producer is much earlier, while producer lookup
            // still verifies that non-view dependencies are not reversed.
            edge.conservative_source = producer == producer_by_tensor.end();
            edge.producer_split = split_id - 1;
            edge.consumer_split = split_id;
            edge.input_id = input_id;
            edge.bytes = ggml_nbytes(src);
            edges.push_back(edge);
            max_bytes = std::max(max_bytes, edge.bytes);
        }
    }
    std::sort(edges.begin(), edges.end(), [](
            const ggml_backend_sched_kimi_async_h2d_edge & lhs,
            const ggml_backend_sched_kimi_async_h2d_edge & rhs) {
        return std::tie(lhs.producer_split, lhs.consumer_split, lhs.input_id) <
            std::tie(rhs.producer_split, rhs.consumer_split, rhs.input_id);
    });
    if ((int) edges.size() != state->EXPECTED_EDGES || device == nullptr ||
            max_bytes == 0) {
        *decline_reason = "non-canonical-edge-count";
        return false;
    }
    std::vector<int> edge_by_split_input(
        sched->n_splits * GGML_SCHED_MAX_SPLIT_INPUTS, -1);
    for (size_t edge_id = 0; edge_id < edges.size(); ++edge_id) {
        const ggml_backend_sched_kimi_async_h2d_edge & edge = edges[edge_id];
        const size_t index =
            edge.consumer_split * GGML_SCHED_MAX_SPLIT_INPUTS + edge.input_id;
        if (edge_by_split_input[index] != -1) {
            *decline_reason = "duplicate-edge";
            return false;
        }
        edge_by_split_input[index] = (int) edge_id;
    }
    state->edges = std::move(edges);
    state->edge_by_split_input = std::move(edge_by_split_input);
    *plan_device = device;
    *staging_capacity = max_bytes;
    return true;
}

static bool ggml_backend_sched_kimi_async_h2d_begin_graph(
        ggml_backend_sched_t sched) {
    ggml_backend_sched_kimi_async_h2d_state * state = sched->kimi_async_h2d;
    if (state == nullptr) {
        return true;
    }
    state->active_graph = false;
    state->edges.clear();
    state->edge_by_split_input.clear();
    if (g_kimi_split_profile_phase != 2 || !sched->decode_phase) {
        return true;
    }
    state->graphs_started++;
    if (state->failed) {
        state->graphs_declined++;
        state->last_decline_reason = "state-poisoned";
        return true;
    }
    const char * decline_reason = "none";
    ggml_backend_dev_t device = nullptr;
    size_t staging_capacity = 0;
    if (!ggml_backend_sched_kimi_async_h2d_build_plan(
                sched, state, &decline_reason, &device, &staging_capacity)) {
        state->graphs_declined++;
        state->last_decline_reason = decline_reason;
        if (!state->decline_logged) {
            GGML_LOG_WARN(
                "%s: async split-input H2D declined reason=%s splits=%d\n",
                __func__, decline_reason, sched->n_splits);
            state->decline_logged = true;
        }
        state->edges.clear();
        state->edge_by_split_input.clear();
        return true;
    }
    if (!ggml_backend_sched_kimi_async_h2d_initialize(
                state, device, staging_capacity)) {
        state->graphs_declined++;
        state->last_decline_reason = "transport-init";
        state->edges.clear();
        state->edge_by_split_input.clear();
        return true;
    }
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!state->tasks.empty() || state->active_tasks != 0) {
            state->ownership_failures++;
            ggml_backend_sched_kimi_async_h2d_fail_locked(
                state, "graph-begin-pending-work");
            return false;
        }
        for (const ggml_backend_sched_kimi_async_h2d_slot & slot :
                state->slots) {
            if (slot.busy) {
                state->ownership_failures++;
                ggml_backend_sched_kimi_async_h2d_fail_locked(
                    state, "graph-begin-busy-slot");
                return false;
            }
        }
    }
    state->graph_epoch++;
    state->next_submit = 0;
    state->producer_frontier = -1;
    state->active_graph = true;
    state->graphs_activated++;
    state->descriptors_planned += state->edges.size();
    state->mapped_sources += std::count_if(
        state->edges.begin(), state->edges.end(), [](
                const ggml_backend_sched_kimi_async_h2d_edge & edge) {
            return !edge.conservative_source;
        });
    state->conservative_sources += std::count_if(
        state->edges.begin(), state->edges.end(), [](
                const ggml_backend_sched_kimi_async_h2d_edge & edge) {
            return edge.conservative_source;
        });
    return true;
}

static int ggml_backend_sched_kimi_async_h2d_free_slot(
        ggml_backend_sched_kimi_async_h2d_state * state) {
    std::lock_guard<std::mutex> lock(state->mutex);
    for (int slot_id = 0; slot_id < state->N_SLOTS; ++slot_id) {
        if (!state->slots[slot_id].busy) {
            return slot_id;
        }
    }
    return -1;
}

static bool ggml_backend_sched_kimi_async_h2d_submit(
        ggml_backend_sched_kimi_async_h2d_state * state,
        size_t edge_id,
        int slot_id) {
    uint64_t generation = 0;
    bool reuse_slot = false;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->failed || edge_id >= state->edges.size() ||
                slot_id < 0 || slot_id >= state->N_SLOTS ||
                state->slots[slot_id].busy) {
            state->ownership_failures++;
            ggml_backend_sched_kimi_async_h2d_fail_locked(
                state, "submit-ownership");
            return false;
        }
        ggml_backend_sched_kimi_async_h2d_edge & edge = state->edges[edge_id];
        if (edge.state != GGML_KIMI_ASYNC_H2D_PLANNED) {
            state->reorder_failures++;
            ggml_backend_sched_kimi_async_h2d_fail_locked(
                state, "submit-reorder");
            return false;
        }
        generation = ++state->next_generation;
        edge.generation = generation;
        edge.slot = slot_id;
        edge.state = GGML_KIMI_ASYNC_H2D_RESERVED;
        ggml_backend_sched_kimi_async_h2d_slot & slot = state->slots[slot_id];
        slot.busy = true;
        slot.generation = generation;
        reuse_slot = slot.has_previous;
        state->descriptors_reserved++;
    }

    {
        std::lock_guard<std::mutex> lock(state->mutex);
        ggml_backend_sched_kimi_async_h2d_edge & edge = state->edges[edge_id];
        state->descriptors_submitted++;
        state->tasks.push_back({
            edge_id, slot_id, generation, edge.src, edge.dst,
            edge.src_backend, edge.bytes, reuse_slot, false,
        });
        state->queue_max = std::max<uint64_t>(
            state->queue_max, state->tasks.size());
        state->work_cv.notify_one();
    }

    const int64_t wait_start_us = ggml_time_us();
    bool blocked = false;
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        blocked = state->edges[edge_id].state == GGML_KIMI_ASYNC_H2D_RESERVED;
        state->state_cv.wait(lock, [&]() {
            return state->failed ||
                state->edges[edge_id].state != GGML_KIMI_ASYNC_H2D_RESERVED;
        });
        if (state->failed) {
            return false;
        }
    }
    if (blocked) {
        const uint64_t wait_us =
            (uint64_t) (ggml_time_us() - wait_start_us);
        state->snapshot_waits++;
        state->snapshot_wait_us += wait_us;
        state->snapshot_wait_max_us = std::max(
            state->snapshot_wait_max_us, wait_us);
    }
    return true;
}

static bool ggml_backend_sched_kimi_async_h2d_pump(
        ggml_backend_sched_kimi_async_h2d_state * state,
        bool block_for_slot) {
    if (!state->active_graph) {
        return true;
    }
    while (state->next_submit < state->edges.size()) {
        const ggml_backend_sched_kimi_async_h2d_edge & edge =
            state->edges[state->next_submit];
        if (edge.producer_split > state->producer_frontier) {
            return true;
        }
        int slot_id = ggml_backend_sched_kimi_async_h2d_free_slot(state);
        if (slot_id < 0 && block_for_slot) {
            std::unique_lock<std::mutex> lock(state->mutex);
            state->state_cv.wait(lock, [&]() {
                if (state->failed) {
                    return true;
                }
                for (const ggml_backend_sched_kimi_async_h2d_slot & slot :
                        state->slots) {
                    if (!slot.busy) {
                        return true;
                    }
                }
                return false;
            });
            if (state->failed) {
                return false;
            }
            lock.unlock();
            slot_id = ggml_backend_sched_kimi_async_h2d_free_slot(state);
        }
        if (slot_id < 0) {
            return true;
        }
        if (!ggml_backend_sched_kimi_async_h2d_submit(
                    state, state->next_submit, slot_id)) {
            return false;
        }
        state->next_submit++;
    }
    return true;
}

static bool ggml_backend_sched_kimi_async_h2d_producer_submitted(
        ggml_backend_sched_t sched,
        int split_id) {
    ggml_backend_sched_kimi_async_h2d_state * state = sched->kimi_async_h2d;
    if (state == nullptr || !state->active_graph) {
        return true;
    }
    if (split_id != state->producer_frontier + 1) {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->reorder_failures++;
        ggml_backend_sched_kimi_async_h2d_fail_locked(
            state, "producer-frontier");
        return false;
    }
    state->producer_frontier = split_id;
    return ggml_backend_sched_kimi_async_h2d_pump(state, false);
}

static int ggml_backend_sched_kimi_async_h2d_edge_id(
        const ggml_backend_sched_kimi_async_h2d_state * state,
        int split_id,
        int input_id) {
    if (!state->active_graph || split_id < 0 || input_id < 0 ||
            split_id >= ggml_backend_sched_kimi_async_h2d_state::EXPECTED_SPLITS ||
            input_id >= GGML_SCHED_MAX_SPLIT_INPUTS) {
        return -1;
    }
    const size_t index =
        split_id * GGML_SCHED_MAX_SPLIT_INPUTS + input_id;
    return index < state->edge_by_split_input.size() ?
        state->edge_by_split_input[index] : -1;
}

static bool ggml_backend_sched_kimi_async_h2d_prepare_input(
        ggml_backend_sched_t sched,
        int split_id,
        int input_id,
        ggml_tensor * src,
        ggml_tensor * dst) {
    ggml_backend_sched_kimi_async_h2d_state * state = sched->kimi_async_h2d;
    const int edge_id = state == nullptr ? -1 :
        ggml_backend_sched_kimi_async_h2d_edge_id(
            state, split_id, input_id);
    if (edge_id < 0) {
        return false;
    }
    if (!ggml_backend_sched_kimi_async_h2d_pump(state, false)) {
        return false;
    }
    int slot_id = -1;
    uint64_t generation = 0;
    ggml_backend_t dst_backend = nullptr;
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        if (state->edges[edge_id].state == GGML_KIMI_ASYNC_H2D_PLANNED) {
            state->reorder_failures++;
            ggml_backend_sched_kimi_async_h2d_fail_locked(
                state, "consumer-edge-not-submitted");
            return false;
        }
        state->state_cv.wait(lock, [&]() {
            return state->failed ||
                state->edges[edge_id].state ==
                    GGML_KIMI_ASYNC_H2D_SOURCE_SNAPSHOTTED;
        });
        if (state->failed) {
            return false;
        }
        ggml_backend_sched_kimi_async_h2d_edge & edge = state->edges[edge_id];
        if (edge.src != src || edge.dst != dst ||
                edge.bytes != ggml_nbytes(src) ||
                edge.bytes != ggml_nbytes(dst) ||
                edge.src_buffer != ggml_backend_sched_kimi_tensor_buffer(src) ||
                edge.dst_buffer != ggml_backend_sched_kimi_tensor_buffer(dst)) {
            state->identity_failures++;
            ggml_backend_sched_kimi_async_h2d_fail_locked(
                state, "consumer-identity");
            return false;
        }
        if (!ggml_are_same_layout(src, dst)) {
            state->layout_failures++;
            ggml_backend_sched_kimi_async_h2d_fail_locked(
                state, "consumer-layout");
            return false;
        }
        slot_id = edge.slot;
        generation = edge.generation;
        dst_backend = edge.dst_backend;
        if (slot_id < 0 || slot_id >= state->N_SLOTS ||
                !state->slots[slot_id].busy ||
                state->slots[slot_id].generation != generation) {
            state->ownership_failures++;
            ggml_backend_sched_kimi_async_h2d_fail_locked(
                state, "consumer-slot-ownership");
            return false;
        }
    }

    // Publish destination ownership at the same consumer frontier where the
    // legacy path synchronized the destination backend.  The worker cannot
    // observe RELEASE_PUBLISHED until this record call has returned.
    ggml_backend_event_record(
        state->slots[slot_id].destination_release, dst_backend);
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        ggml_backend_sched_kimi_async_h2d_edge & edge = state->edges[edge_id];
        if (edge.state != GGML_KIMI_ASYNC_H2D_SOURCE_SNAPSHOTTED ||
                edge.generation != generation || edge.slot != slot_id) {
            state->stale_generation_failures++;
            ggml_backend_sched_kimi_async_h2d_fail_locked(
                state, "consumer-release-generation");
            return false;
        }
        edge.state = GGML_KIMI_ASYNC_H2D_RELEASE_PUBLISHED;
        state->release_records++;
        state->tasks.push_back({
            (size_t) edge_id, slot_id, generation, edge.src, edge.dst,
            edge.src_backend, edge.bytes, false, true,
        });
        state->queue_max = std::max<uint64_t>(
            state->queue_max, state->tasks.size());
        state->work_cv.notify_one();
    }

    {
        std::unique_lock<std::mutex> lock(state->mutex);
        state->state_cv.wait(lock, [&]() {
            return state->failed ||
                state->edges[edge_id].state ==
                    GGML_KIMI_ASYNC_H2D_READY_PUBLISHED;
        });
        if (state->failed) {
            return false;
        }
    }
    ggml_backend_event_wait(dst_backend, state->slots[slot_id].input_ready);
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        ggml_backend_sched_kimi_async_h2d_edge & edge = state->edges[edge_id];
        if (edge.state != GGML_KIMI_ASYNC_H2D_READY_PUBLISHED ||
                edge.generation != generation || edge.slot != slot_id ||
                !state->slots[slot_id].busy ||
                state->slots[slot_id].generation != generation) {
            state->stale_generation_failures++;
            ggml_backend_sched_kimi_async_h2d_fail_locked(
                state, "consumer-wait-generation");
            return false;
        }
        edge.state = GGML_KIMI_ASYNC_H2D_CONSUMER_QUEUED;
        edge.slot = -1;
        state->slots[slot_id].busy = false;
        state->slots[slot_id].has_previous = true;
        state->ready_waits++;
        state->consumer_queued++;
        state->state_cv.notify_all();
    }
    return ggml_backend_sched_kimi_async_h2d_pump(state, false);
}

static bool ggml_backend_sched_kimi_async_h2d_consumer_begin(
        ggml_backend_sched_t sched,
        int split_id) {
    ggml_backend_sched_kimi_async_h2d_state * state = sched->kimi_async_h2d;
    if (state == nullptr || !state->active_graph) {
        return true;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    for (const ggml_backend_sched_kimi_async_h2d_edge & edge : state->edges) {
        if (edge.consumer_split == split_id &&
                edge.state != GGML_KIMI_ASYNC_H2D_CONSUMER_QUEUED) {
            state->reorder_failures++;
            ggml_backend_sched_kimi_async_h2d_fail_locked(
                state, "consumer-launch-before-wait");
            return false;
        }
    }
    return true;
}

static bool ggml_backend_sched_kimi_async_h2d_consumer_end(
        ggml_backend_sched_t sched,
        int split_id) {
    ggml_backend_sched_kimi_async_h2d_state * state = sched->kimi_async_h2d;
    if (state == nullptr || !state->active_graph) {
        return true;
    }
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        for (ggml_backend_sched_kimi_async_h2d_edge & edge : state->edges) {
            if (edge.consumer_split != split_id) {
                continue;
            }
            if (edge.state != GGML_KIMI_ASYNC_H2D_CONSUMER_QUEUED ||
                    edge.slot != -1) {
                state->reorder_failures++;
                ggml_backend_sched_kimi_async_h2d_fail_locked(
                    state, "consumer-retire-reorder");
                return false;
            }
            edge.state = GGML_KIMI_ASYNC_H2D_RETIRED;
            state->descriptors_retired++;
        }
        state->state_cv.notify_all();
    }
    return ggml_backend_sched_kimi_async_h2d_pump(state, false);
}

static bool ggml_backend_sched_kimi_async_h2d_finish_graph(
        ggml_backend_sched_t sched) {
    ggml_backend_sched_kimi_async_h2d_state * state = sched->kimi_async_h2d;
    if (state == nullptr || !state->active_graph) {
        return true;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    bool closed = !state->failed &&
        state->next_submit == state->edges.size() &&
        state->tasks.empty() && state->active_tasks == 0;
    for (const ggml_backend_sched_kimi_async_h2d_edge & edge : state->edges) {
        closed = closed && edge.state == GGML_KIMI_ASYNC_H2D_RETIRED;
    }
    for (const ggml_backend_sched_kimi_async_h2d_slot & slot : state->slots) {
        closed = closed && !slot.busy;
    }
    if (!closed) {
        state->ownership_failures++;
        ggml_backend_sched_kimi_async_h2d_fail_locked(
            state, "graph-ledger-open");
        return false;
    }
    state->graphs_completed++;
    state->active_graph = false;
    state->edges.clear();
    state->edge_by_split_input.clear();
    return true;
}

static bool ggml_backend_sched_kimi_async_h2d_drain(
        ggml_backend_sched_kimi_async_h2d_state * state,
        bool poison_active) {
    if (state == nullptr || !state->worker_started) {
        return state == nullptr || !state->active_graph;
    }
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        if (poison_active && state->active_graph) {
            state->graphs_poisoned++;
            ggml_backend_sched_kimi_async_h2d_fail_locked(
                state, "graph-aborted");
        }
        state->state_cv.wait(lock, [&]() {
            return state->tasks.empty() && state->active_tasks == 0;
        });
    }
    ggml_backend_synchronize(state->backend);
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->active_graph) {
            for (ggml_backend_sched_kimi_async_h2d_edge & edge : state->edges) {
                if (edge.state != GGML_KIMI_ASYNC_H2D_RETIRED) {
                    edge.state = GGML_KIMI_ASYNC_H2D_POISONED;
                }
            }
            for (ggml_backend_sched_kimi_async_h2d_slot & slot : state->slots) {
                slot.busy = false;
            }
            state->active_graph = false;
            state->edges.clear();
            state->edge_by_split_input.clear();
        }
    }
    return !state->failed;
}

static void ggml_backend_sched_kimi_async_h2d_shutdown(
        ggml_backend_sched_kimi_async_h2d_state * state) {
    if (state == nullptr) {
        return;
    }
    (void) ggml_backend_sched_kimi_async_h2d_drain(state, true);
    if (state->worker_started) {
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->stop = true;
            state->work_cv.notify_all();
        }
        state->worker.join();
        state->worker_started = false;
    }
    int slots_free = 0;
    for (const ggml_backend_sched_kimi_async_h2d_slot & slot : state->slots) {
        slots_free += !slot.busy;
    }
    const bool guards_closed = !state->failed &&
        state->graphs_activated == state->graphs_completed &&
        state->descriptors_planned == state->descriptors_reserved &&
        state->descriptors_planned ==
            state->mapped_sources + state->conservative_sources &&
        state->descriptors_reserved == state->descriptors_submitted &&
        state->descriptors_submitted == state->source_syncs &&
        state->source_syncs == state->snapshot_calls &&
        state->snapshot_calls == state->release_records &&
        state->release_records == state->release_waits &&
        state->release_waits == state->transfer_enqueued &&
        state->transfer_enqueued == state->ready_records &&
        state->ready_records == state->ready_published &&
        state->ready_published == state->ready_waits &&
        state->ready_waits == state->consumer_queued &&
        state->consumer_queued == state->descriptors_retired &&
        state->h2d_calls == state->snapshot_calls &&
        state->h2d_bytes == state->snapshot_bytes &&
        state->tasks.empty() && state->active_tasks == 0 &&
        slots_free == state->N_SLOTS;
    GGML_LOG_INFO(
        "%s: async_split_input_h2d default_off=0 graphs_started=%llu "
        "activated=%llu declined=%llu completed=%llu poisoned=%llu "
        "planned=%llu mapped_sources=%llu conservative_sources=%llu "
        "reserved=%llu submitted=%llu source_syncs=%llu "
        "snapshots=%llu snapshot_bytes=%llu snapshot_waits=%llu "
        "snapshot_wait_us=%llu snapshot_wait_max_us=%llu "
        "release_records=%llu release_waits=%llu transfer_enqueued=%llu "
        "ready_records=%llu ready_published=%llu ready_waits=%llu "
        "h2d_calls=%llu h2d_bytes=%llu consumer_queued=%llu retired=%llu "
        "reuse_host_waits=%llu worker_wakeups=%llu queue_max=%llu "
        "slots_free=%d stale_failures=%llu identity_failures=%llu "
        "layout_failures=%llu ownership_failures=%llu reorder_failures=%llu "
        "init_failures=%llu failed=%d guards_closed=%d failure_reason=%s "
        "decline_reason=%s\n",
        __func__,
        (unsigned long long) state->graphs_started,
        (unsigned long long) state->graphs_activated,
        (unsigned long long) state->graphs_declined,
        (unsigned long long) state->graphs_completed,
        (unsigned long long) state->graphs_poisoned,
        (unsigned long long) state->descriptors_planned,
        (unsigned long long) state->mapped_sources,
        (unsigned long long) state->conservative_sources,
        (unsigned long long) state->descriptors_reserved,
        (unsigned long long) state->descriptors_submitted,
        (unsigned long long) state->source_syncs,
        (unsigned long long) state->snapshot_calls,
        (unsigned long long) state->snapshot_bytes,
        (unsigned long long) state->snapshot_waits,
        (unsigned long long) state->snapshot_wait_us,
        (unsigned long long) state->snapshot_wait_max_us,
        (unsigned long long) state->release_records,
        (unsigned long long) state->release_waits,
        (unsigned long long) state->transfer_enqueued,
        (unsigned long long) state->ready_records,
        (unsigned long long) state->ready_published,
        (unsigned long long) state->ready_waits,
        (unsigned long long) state->h2d_calls,
        (unsigned long long) state->h2d_bytes,
        (unsigned long long) state->consumer_queued,
        (unsigned long long) state->descriptors_retired,
        (unsigned long long) state->reuse_host_waits,
        (unsigned long long) state->worker_wakeups,
        (unsigned long long) state->queue_max,
        slots_free,
        (unsigned long long) state->stale_generation_failures,
        (unsigned long long) state->identity_failures,
        (unsigned long long) state->layout_failures,
        (unsigned long long) state->ownership_failures,
        (unsigned long long) state->reorder_failures,
        (unsigned long long) state->init_failures,
        state->failed ? 1 : 0,
        guards_closed ? 1 : 0,
        state->failure_reason,
        state->last_decline_reason);
    ggml_backend_sched_kimi_async_h2d_release_resources(state);
}

static void ggml_backend_sched_kimi_async_ids_fail_locked(
        ggml_backend_sched_kimi_async_ids_state * state,
        const char * reason) {
    state->failed = true;
    state->failure_reason = reason;
    state->state_cv.notify_all();
    state->work_cv.notify_all();
}

static void ggml_backend_sched_kimi_async_ids_worker(
        ggml_backend_sched_kimi_async_ids_state * state) {
    while (true) {
        ggml_backend_sched_kimi_async_ids_task task;
        ggml_tensor * ids_tensor = nullptr;
        int64_t n_expert = 0;
        int64_t ne0 = 0;
        int64_t ne1 = 0;
        size_t nb0 = 0;
        size_t nb1 = 0;
        size_t bytes = 0;
        {
            std::unique_lock<std::mutex> lock(state->mutex);
            state->work_cv.wait(lock, [&]() {
                return state->stop || !state->tasks.empty();
            });
            if (state->stop && state->tasks.empty()) {
                break;
            }
            task = state->tasks.front();
            state->tasks.pop_front();
            state->active_tasks++;
            state->worker_wakeups++;
            if (task.plan >= state->plans.size() || task.slot < 0 ||
                    task.slot >= state->N_SLOTS) {
                state->ownership_failures++;
                ggml_backend_sched_kimi_async_ids_fail_locked(
                    state, "worker-task-identity");
                state->active_tasks--;
                state->state_cv.notify_all();
                continue;
            }
            ggml_backend_sched_kimi_async_ids_plan & plan =
                state->plans[task.plan];
            ggml_backend_sched_kimi_async_ids_slot & slot =
                state->slots[task.slot];
            if (!slot.busy || slot.generation != task.generation ||
                    plan.generation != task.generation ||
                    plan.slot != task.slot ||
                    plan.state != GGML_KIMI_ASYNC_IDS_SOURCE_PUBLISHED) {
                state->stale_generation_failures++;
                ggml_backend_sched_kimi_async_ids_fail_locked(
                    state, "worker-stale-generation");
                state->active_tasks--;
                state->state_cv.notify_all();
                continue;
            }
            plan.state = GGML_KIMI_ASYNC_IDS_TRANSFER_ENQUEUED;
            state->transfer_enqueued++;
            ids_tensor = plan.ids_tensor;
            n_expert = plan.n_expert;
            ne0 = plan.ne0;
            ne1 = plan.ne1;
            nb0 = plan.nb0;
            nb1 = plan.nb1;
            bytes = plan.bytes;
        }

        ggml_backend_sched_kimi_async_ids_slot & slot =
            state->slots[task.slot];
        ggml_backend_event_wait(state->backend, slot.producer_ready);
        state->producer_waits++;
        ggml_backend_tensor_get_async(
            state->backend,
            ids_tensor,
            ggml_backend_buffer_get_base(slot.staging),
            0,
            bytes);
        state->d2h_calls++;
        state->d2h_bytes += bytes;
        ggml_backend_event_record(slot.d2h_done, state->backend);
        state->done_records++;
        ggml_backend_event_synchronize(slot.d2h_done);
        state->done_host_waits++;

        std::vector<int32_t> parsed(bytes / sizeof(int32_t));
        memcpy(
            parsed.data(), ggml_backend_buffer_get_base(slot.staging), bytes);
        std::vector<ggml_bitset_t> parsed_used(
            ggml_bitset_size(n_expert));
        std::vector<int32_t> parsed_counts((size_t) n_expert);
        bool parse_ok = nb0 % sizeof(int32_t) == 0 &&
            nb1 % sizeof(int32_t) == 0;
        uint64_t parsed_elements = 0;
        for (int64_t i1 = 0; parse_ok && i1 < ne1; ++i1) {
            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                const size_t idx =
                    (size_t) i1 * nb1 / sizeof(int32_t) +
                    (size_t) i0 * nb0 / sizeof(int32_t);
                if (idx >= parsed.size()) {
                    parse_ok = false;
                    break;
                }
                const int32_t id = parsed[idx];
                if (id < 0 || id >= n_expert) {
                    parse_ok = false;
                    break;
                }
                ggml_bitset_set(parsed_used.data(), id);
                parsed_counts[id]++;
                parsed_elements++;
            }
        }

        {
            std::lock_guard<std::mutex> lock(state->mutex);
            ggml_backend_sched_kimi_async_ids_plan & plan =
                state->plans[task.plan];
            if (!parse_ok) {
                state->range_failures++;
                ggml_backend_sched_kimi_async_ids_fail_locked(
                    state, "worker-parse-range");
            } else if (!slot.busy ||
                    slot.generation != task.generation ||
                    plan.generation != task.generation ||
                    plan.slot != task.slot ||
                    plan.state != GGML_KIMI_ASYNC_IDS_TRANSFER_ENQUEUED) {
                state->stale_generation_failures++;
                ggml_backend_sched_kimi_async_ids_fail_locked(
                    state, "worker-completion-generation");
            } else {
                slot.ids = std::move(parsed);
                slot.used_ids = std::move(parsed_used);
                slot.id_counts = std::move(parsed_counts);
                plan.state = GGML_KIMI_ASYNC_IDS_HOST_READY;
                state->plans_ready++;
                state->parse_calls++;
                state->parsed_elements += parsed_elements;
            }
            state->active_tasks--;
            state->state_cv.notify_all();
        }
    }
}

static void ggml_backend_sched_kimi_async_ids_release_resources(
        ggml_backend_sched_kimi_async_ids_state * state) {
    for (ggml_backend_sched_kimi_async_ids_slot & slot : state->slots) {
        ggml_backend_event_free(slot.producer_ready);
        ggml_backend_event_free(slot.d2h_done);
        ggml_backend_buffer_free(slot.staging);
        slot.producer_ready = nullptr;
        slot.d2h_done = nullptr;
        slot.staging = nullptr;
        slot.ids.clear();
        slot.used_ids.clear();
        slot.id_counts.clear();
        slot.busy = false;
    }
    ggml_backend_free(state->backend);
    state->backend = nullptr;
    state->device = nullptr;
    state->host_buft = nullptr;
    state->staging_capacity = 0;
}

static bool ggml_backend_sched_kimi_async_ids_initialize(
        ggml_backend_sched_kimi_async_ids_state * state,
        ggml_backend_dev_t device,
        size_t staging_capacity) {
    if (state->worker_started) {
        return state->device == device &&
            state->staging_capacity >= staging_capacity;
    }
    state->device = device;
    state->host_buft = ggml_backend_dev_host_buffer_type(device);
    state->backend = ggml_backend_dev_init(device, nullptr);
    if (state->host_buft == nullptr || state->backend == nullptr ||
            state->backend->iface.get_tensor_async == nullptr ||
            state->backend->iface.event_wait == nullptr ||
            state->backend->iface.event_record == nullptr) {
        state->init_failures++;
        ggml_backend_sched_kimi_async_ids_release_resources(state);
        return false;
    }
    state->staging_capacity = staging_capacity;
    for (ggml_backend_sched_kimi_async_ids_slot & slot : state->slots) {
        slot.staging = ggml_backend_buft_alloc_buffer(
            state->host_buft, staging_capacity);
        slot.producer_ready = ggml_backend_event_new(device);
        slot.d2h_done = ggml_backend_event_new(device);
        if (slot.staging == nullptr || slot.producer_ready == nullptr ||
                slot.d2h_done == nullptr ||
                !ggml_backend_buffer_is_host(slot.staging) ||
                ggml_backend_buffer_get_base(slot.staging) == nullptr) {
            state->init_failures++;
            ggml_backend_sched_kimi_async_ids_release_resources(state);
            return false;
        }
    }
    try {
        state->worker = std::thread(
            ggml_backend_sched_kimi_async_ids_worker, state);
        state->worker_started = true;
    } catch (...) {
        state->init_failures++;
        ggml_backend_sched_kimi_async_ids_release_resources(state);
        return false;
    }
    GGML_LOG_INFO(
        "%s: async expert-id preparation worker enabled device=%s slots=%d "
        "staging_capacity=%zu\n",
        __func__, ggml_backend_dev_name(device), state->N_SLOTS,
        staging_capacity);
    return true;
}

static bool ggml_backend_sched_kimi_async_ids_build_plan(
        ggml_backend_sched_t sched,
        ggml_backend_sched_kimi_async_ids_state * state,
        const char ** decline_reason,
        ggml_backend_dev_t * plan_device,
        size_t * staging_capacity) {
    if (sched->n_copies != 1) {
        *decline_reason = "scheduler-copies";
        return false;
    }
    if (sched->callback_eval != nullptr) {
        *decline_reason = "callback-eval";
        return false;
    }
    if (sched->n_splits != state->EXPECTED_SPLITS) {
        *decline_reason = "non-canonical-split-count";
        return false;
    }

    std::map<const ggml_tensor *, int> producer_by_tensor;
    for (int split_id = 0; split_id < sched->n_splits; ++split_id) {
        const ggml_backend_sched_split & split = sched->splits[split_id];
        for (int node_id = 0; node_id < split.graph.n_nodes; ++node_id) {
            ggml_tensor * node = split.graph.nodes[node_id];
            auto inserted = producer_by_tensor.emplace(node, split_id);
            if (!inserted.second && inserted.first->second != split_id) {
                *decline_reason = "duplicate-producer";
                return false;
            }
        }
    }

    std::vector<ggml_backend_sched_kimi_async_ids_plan> plans;
    ggml_tensor * previous_ids = nullptr;
    int64_t previous_n_expert = -1;
    ggml_backend_dev_t device = nullptr;
    size_t max_bytes = 0;
    for (int split_id = 0; split_id < sched->n_splits; ++split_id) {
        ggml_backend_sched_split & split = sched->splits[split_id];
        ggml_backend_t split_backend = sched->backends[split.backend_id];
        for (int input_id = 0; input_id < split.n_inputs; ++input_id) {
            ggml_tensor * input = split.inputs[input_id];
            ggml_backend_buffer_t input_buffer =
                ggml_backend_sched_kimi_tensor_buffer(input);
            if (input_buffer == nullptr ||
                    ggml_backend_buffer_get_usage(input_buffer) !=
                        GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
                    !ggml_backend_buffer_is_host(input_buffer)) {
                continue;
            }
            ggml_tensor * input_cpy = tensor_copy(
                input, split.backend_id, sched->cur_copy);
            ggml_tensor * node = nullptr;
            for (int node_id = 0; node_id < split.graph.n_nodes; ++node_id) {
                ggml_tensor * candidate = split.graph.nodes[node_id];
                if (candidate->op == GGML_OP_MUL_MAT_ID &&
                        candidate->src[0] == input_cpy) {
                    node = candidate;
                    break;
                }
            }
            if (node == nullptr) {
                continue;
            }

            const int64_t n_expert = input->ne[2];
            ggml_tensor * ids_tensor = node->src[2];
            ggml_backend_t ids_backend = split_backend;
            for (int i = input_id + 1; i < split.n_inputs; ++i) {
                if (ids_tensor == tensor_copy(
                        split.inputs[i], split.backend_id,
                        sched->cur_copy)) {
                    ids_tensor = split.inputs[i];
                    ids_backend =
                        ggml_backend_sched_get_tensor_backend(
                            sched, split.inputs[i]);
                    break;
                }
            }
            if (ids_tensor == previous_ids &&
                    n_expert == previous_n_expert) {
                continue;
            }
            previous_ids = ids_tensor;
            previous_n_expert = n_expert;

            auto producer = producer_by_tensor.find(ids_tensor);
            const ggml_tensor * view_root =
                ggml_backend_sched_kimi_view_root(ids_tensor);
            auto root_producer = producer_by_tensor.find(view_root);
            if (root_producer != producer_by_tensor.end() &&
                    (producer == producer_by_tensor.end() ||
                     root_producer->second < producer->second)) {
                producer = root_producer;
            }
            if (producer == producer_by_tensor.end() ||
                    producer->second >= split_id) {
                *decline_reason = "ids-producer-order";
                return false;
            }
            ggml_backend_dev_t ids_device =
                ggml_backend_get_device(ids_backend);
            ggml_backend_buffer_t ids_buffer =
                ggml_backend_sched_kimi_tensor_buffer(ids_tensor);
            const size_t bytes = ggml_nbytes(ids_tensor);
            if (ids_tensor->type != GGML_TYPE_I32 ||
                    ids_tensor->ne[2] != 1 || ids_tensor->ne[3] != 1 ||
                    n_expert <= 0 || n_expert > INT_MAX ||
                    bytes != state->EXPECTED_BYTES_PER_PLAN ||
                    ids_tensor->data == nullptr || ids_buffer == nullptr ||
                    ggml_backend_dev_type(ids_device) !=
                        GGML_BACKEND_DEVICE_TYPE_GPU ||
                    ids_backend->iface.event_record == nullptr ||
                    ids_device->iface.event_new == nullptr) {
                *decline_reason = "ids-layout-or-capability";
                return false;
            }
            if (device != nullptr && device != ids_device) {
                *decline_reason = "ids-cross-device";
                return false;
            }
            device = ids_device;
            ggml_backend_sched_kimi_async_ids_plan plan;
            plan.ids_tensor = ids_tensor;
            plan.ids_backend = ids_backend;
            plan.ids_buffer = ids_buffer;
            plan.producer_split = producer->second;
            plan.consumer_split = split_id;
            plan.consumer_input = input_id;
            plan.n_expert = n_expert;
            plan.ne0 = ids_tensor->ne[0];
            plan.ne1 = ids_tensor->ne[1];
            plan.nb0 = ids_tensor->nb[0];
            plan.nb1 = ids_tensor->nb[1];
            plan.bytes = bytes;
            plans.push_back(plan);
            max_bytes = std::max(max_bytes, bytes);
        }
    }
    if ((int) plans.size() != state->EXPECTED_PLANS || device == nullptr ||
            max_bytes != state->EXPECTED_BYTES_PER_PLAN) {
        *decline_reason = "non-canonical-plan-count";
        return false;
    }
    if (!std::is_sorted(
            plans.begin(), plans.end(), [](
                const ggml_backend_sched_kimi_async_ids_plan & lhs,
                const ggml_backend_sched_kimi_async_ids_plan & rhs) {
                return std::tie(lhs.producer_split, lhs.consumer_split,
                                lhs.consumer_input) <
                    std::tie(rhs.producer_split, rhs.consumer_split,
                             rhs.consumer_input);
            })) {
        *decline_reason = "non-monotonic-plan-order";
        return false;
    }
    std::vector<int> plan_by_split_input(
        sched->n_splits * GGML_SCHED_MAX_SPLIT_INPUTS, -1);
    for (size_t plan_id = 0; plan_id < plans.size(); ++plan_id) {
        const ggml_backend_sched_kimi_async_ids_plan & plan = plans[plan_id];
        const size_t index =
            (size_t) plan.consumer_split * GGML_SCHED_MAX_SPLIT_INPUTS +
            (size_t) plan.consumer_input;
        if (plan_by_split_input[index] != -1) {
            *decline_reason = "duplicate-consumer";
            return false;
        }
        plan_by_split_input[index] = (int) plan_id;
    }
    state->plans = std::move(plans);
    state->plan_by_split_input = std::move(plan_by_split_input);
    *plan_device = device;
    *staging_capacity = max_bytes;
    return true;
}

static bool ggml_backend_sched_kimi_async_ids_begin_graph(
        ggml_backend_sched_t sched) {
    ggml_backend_sched_kimi_async_ids_state * state = sched->kimi_async_ids;
    if (state == nullptr) return true;
    state->active_graph = false;
    state->plans.clear();
    state->plan_by_split_input.clear();
    if (g_kimi_split_profile_phase != 2 ||
            !sched->decode_phase) {
        return true;
    }
    state->graphs_started++;
    if (state->failed) {
        state->graphs_declined++;
        state->last_decline_reason = "state-poisoned";
        return true;
    }
    const char * decline_reason = "none";
    ggml_backend_dev_t device = nullptr;
    size_t capacity = 0;
    if (!ggml_backend_sched_kimi_async_ids_build_plan(
                sched, state, &decline_reason, &device, &capacity)) {
        state->graphs_declined++;
        state->last_decline_reason = decline_reason;
        if (!state->decline_logged) {
            GGML_LOG_WARN(
                "%s: async expert-id preparation declined reason=%s "
                "splits=%d\n",
                __func__, decline_reason, sched->n_splits);
            state->decline_logged = true;
        }
        state->plans.clear();
        state->plan_by_split_input.clear();
        return true;
    }
    if (!ggml_backend_sched_kimi_async_ids_initialize(
                state, device, capacity)) {
        state->graphs_declined++;
        state->last_decline_reason = "transport-init";
        state->plans.clear();
        state->plan_by_split_input.clear();
        return true;
    }
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!state->tasks.empty() || state->active_tasks != 0) {
            state->ownership_failures++;
            ggml_backend_sched_kimi_async_ids_fail_locked(
                state, "graph-begin-pending-work");
            return false;
        }
        for (const ggml_backend_sched_kimi_async_ids_slot & slot :
                state->slots) {
            if (slot.busy) {
                state->ownership_failures++;
                ggml_backend_sched_kimi_async_ids_fail_locked(
                    state, "graph-begin-busy-slot");
                return false;
            }
        }
    }
    state->graph_epoch++;
    state->next_submit = 0;
    state->producer_frontier = -1;
    state->active_graph = true;
    state->graphs_activated++;
    state->plans_planned += state->plans.size();
    return true;
}

static int ggml_backend_sched_kimi_async_ids_free_slot(
        ggml_backend_sched_kimi_async_ids_state * state) {
    std::lock_guard<std::mutex> lock(state->mutex);
    for (int slot_id = 0; slot_id < state->N_SLOTS; ++slot_id) {
        if (!state->slots[slot_id].busy) return slot_id;
    }
    return -1;
}

static bool ggml_backend_sched_kimi_async_ids_submit(
        ggml_backend_sched_kimi_async_ids_state * state,
        size_t plan_id,
        int slot_id) {
    uint64_t generation = 0;
    ggml_backend_t ids_backend = nullptr;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->failed || plan_id >= state->plans.size() ||
                slot_id < 0 || slot_id >= state->N_SLOTS ||
                state->slots[slot_id].busy) {
            state->ownership_failures++;
            ggml_backend_sched_kimi_async_ids_fail_locked(
                state, "submit-ownership");
            return false;
        }
        ggml_backend_sched_kimi_async_ids_plan & plan =
            state->plans[plan_id];
        if (plan.state != GGML_KIMI_ASYNC_IDS_PLANNED) {
            state->reorder_failures++;
            ggml_backend_sched_kimi_async_ids_fail_locked(
                state, "submit-reorder");
            return false;
        }
        generation = ++state->next_generation;
        plan.generation = generation;
        plan.slot = slot_id;
        plan.state = GGML_KIMI_ASYNC_IDS_RESERVED;
        ggml_backend_sched_kimi_async_ids_slot & slot =
            state->slots[slot_id];
        slot.busy = true;
        slot.generation = generation;
        ids_backend = plan.ids_backend;
        state->plans_reserved++;
    }
    ggml_backend_event_record(
        state->slots[slot_id].producer_ready, ids_backend);
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        ggml_backend_sched_kimi_async_ids_plan & plan =
            state->plans[plan_id];
        ggml_backend_sched_kimi_async_ids_slot & slot =
            state->slots[slot_id];
        if (state->failed || plan.generation != generation ||
                plan.state != GGML_KIMI_ASYNC_IDS_RESERVED ||
                !slot.busy || slot.generation != generation) {
            state->stale_generation_failures++;
            ggml_backend_sched_kimi_async_ids_fail_locked(
                state, "submit-publish-generation");
            return false;
        }
        plan.state = GGML_KIMI_ASYNC_IDS_SOURCE_PUBLISHED;
        state->producer_records++;
        state->plans_submitted++;
        state->tasks.push_back({plan_id, slot_id, generation});
        state->queue_max = std::max<uint64_t>(
            state->queue_max, state->tasks.size());
        state->work_cv.notify_one();
    }
    return true;
}

static bool ggml_backend_sched_kimi_async_ids_pump(
        ggml_backend_sched_kimi_async_ids_state * state) {
    if (!state->active_graph) return true;
    while (state->next_submit < state->plans.size()) {
        const ggml_backend_sched_kimi_async_ids_plan & plan =
            state->plans[state->next_submit];
        if (plan.producer_split > state->producer_frontier) return true;
        const int slot_id =
            ggml_backend_sched_kimi_async_ids_free_slot(state);
        if (slot_id < 0) return true;
        if (!ggml_backend_sched_kimi_async_ids_submit(
                    state, state->next_submit, slot_id)) {
            return false;
        }
        state->next_submit++;
    }
    return true;
}

static bool ggml_backend_sched_kimi_async_ids_producer_submitted(
        ggml_backend_sched_t sched,
        int split_id) {
    ggml_backend_sched_kimi_async_ids_state * state = sched->kimi_async_ids;
    if (state == nullptr || !state->active_graph) return true;
    if (split_id != state->producer_frontier + 1) {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->reorder_failures++;
        ggml_backend_sched_kimi_async_ids_fail_locked(
            state, "producer-frontier");
        return false;
    }
    state->producer_frontier = split_id;
    if (!ggml_backend_sched_kimi_async_ids_pump(state)) return false;
    if (state->next_submit < state->plans.size() &&
            state->plans[state->next_submit].producer_split <= split_id) {
        state->slot_deferrals++;
    }
    return true;
}

static int ggml_backend_sched_kimi_async_ids_plan_id(
        const ggml_backend_sched_kimi_async_ids_state * state,
        int split_id,
        int input_id) {
    if (!state || !state->active_graph || split_id < 0 || input_id < 0 ||
            input_id >= GGML_SCHED_MAX_SPLIT_INPUTS) {
        return -1;
    }
    const size_t index =
        (size_t) split_id * GGML_SCHED_MAX_SPLIT_INPUTS + input_id;
    return index < state->plan_by_split_input.size() ?
        state->plan_by_split_input[index] : -1;
}

static bool ggml_backend_sched_kimi_async_ids_consume(
        ggml_backend_sched_t sched,
        int split_id,
        int input_id,
        ggml_tensor * ids_tensor,
        ggml_backend_t ids_backend,
        int64_t n_expert,
        std::vector<int32_t> & ids,
        std::vector<ggml_bitset_t> & used_ids,
        std::vector<int32_t> & id_counts) {
    ggml_backend_sched_kimi_async_ids_state * state = sched->kimi_async_ids;
    const int plan_id = ggml_backend_sched_kimi_async_ids_plan_id(
        state, split_id, input_id);
    if (plan_id < 0) return false;
    if (!ggml_backend_sched_kimi_async_ids_pump(state)) return false;
    bool blocked = false;
    int slot_id = -1;
    uint64_t generation = 0;
    const int64_t wait_start_us = ggml_time_us();
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        if (state->plans[plan_id].state == GGML_KIMI_ASYNC_IDS_PLANNED) {
            state->reorder_failures++;
            ggml_backend_sched_kimi_async_ids_fail_locked(
                state, "consumer-plan-not-submitted");
            return false;
        }
        blocked = state->plans[plan_id].state !=
            GGML_KIMI_ASYNC_IDS_HOST_READY;
        state->state_cv.wait(lock, [&]() {
            return state->failed || state->plans[plan_id].state ==
                GGML_KIMI_ASYNC_IDS_HOST_READY;
        });
        if (state->failed) return false;
        ggml_backend_sched_kimi_async_ids_plan & plan =
            state->plans[plan_id];
        if (plan.ids_tensor != ids_tensor ||
                plan.ids_backend != ids_backend ||
                plan.ids_buffer !=
                    ggml_backend_sched_kimi_tensor_buffer(ids_tensor) ||
                plan.n_expert != n_expert ||
                plan.bytes != ggml_nbytes(ids_tensor)) {
            state->identity_failures++;
            ggml_backend_sched_kimi_async_ids_fail_locked(
                state, "consumer-identity");
            return false;
        }
        if (plan.ne0 != ids_tensor->ne[0] ||
                plan.ne1 != ids_tensor->ne[1] ||
                plan.nb0 != ids_tensor->nb[0] ||
                plan.nb1 != ids_tensor->nb[1]) {
            state->layout_failures++;
            ggml_backend_sched_kimi_async_ids_fail_locked(
                state, "consumer-layout");
            return false;
        }
        slot_id = plan.slot;
        generation = plan.generation;
        if (slot_id < 0 || slot_id >= state->N_SLOTS ||
                !state->slots[slot_id].busy ||
                state->slots[slot_id].generation != generation) {
            state->ownership_failures++;
            ggml_backend_sched_kimi_async_ids_fail_locked(
                state, "consumer-slot-ownership");
            return false;
        }
        ids = state->slots[slot_id].ids;
        used_ids = state->slots[slot_id].used_ids;
        id_counts = state->slots[slot_id].id_counts;
        plan.state = GGML_KIMI_ASYNC_IDS_CONSUMED;
        state->plans_consumed++;
        plan.state = GGML_KIMI_ASYNC_IDS_RETIRED;
        plan.slot = -1;
        state->slots[slot_id].busy = false;
        state->slots[slot_id].ids.clear();
        state->slots[slot_id].used_ids.clear();
        state->slots[slot_id].id_counts.clear();
        state->plans_retired++;
        state->state_cv.notify_all();
    }
    if (blocked) {
        const uint64_t wait_us = (uint64_t) (ggml_time_us() - wait_start_us);
        state->consumer_waits++;
        state->consumer_wait_us += wait_us;
        state->consumer_wait_max_us = std::max(
            state->consumer_wait_max_us, wait_us);
    }
    return ggml_backend_sched_kimi_async_ids_pump(state);
}

static bool ggml_backend_sched_kimi_async_ids_finish_graph(
        ggml_backend_sched_t sched) {
    ggml_backend_sched_kimi_async_ids_state * state = sched->kimi_async_ids;
    if (state == nullptr || !state->active_graph) return true;
    std::lock_guard<std::mutex> lock(state->mutex);
    bool closed = !state->failed &&
        state->next_submit == state->plans.size() &&
        state->tasks.empty() && state->active_tasks == 0;
    for (const ggml_backend_sched_kimi_async_ids_plan & plan : state->plans) {
        closed = closed && plan.state == GGML_KIMI_ASYNC_IDS_RETIRED;
    }
    for (const ggml_backend_sched_kimi_async_ids_slot & slot : state->slots) {
        closed = closed && !slot.busy;
    }
    if (!closed) {
        state->ownership_failures++;
        ggml_backend_sched_kimi_async_ids_fail_locked(
            state, "graph-ledger-open");
        return false;
    }
    state->graphs_completed++;
    state->active_graph = false;
    state->plans.clear();
    state->plan_by_split_input.clear();
    return true;
}

static bool ggml_backend_sched_kimi_async_ids_drain(
        ggml_backend_sched_kimi_async_ids_state * state,
        bool poison_active) {
    if (state == nullptr || !state->worker_started) {
        return state == nullptr || !state->active_graph;
    }
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        if (poison_active && state->active_graph) {
            state->graphs_poisoned++;
            ggml_backend_sched_kimi_async_ids_fail_locked(
                state, "graph-aborted");
        }
        state->state_cv.wait(lock, [&]() {
            return state->tasks.empty() && state->active_tasks == 0;
        });
    }
    ggml_backend_synchronize(state->backend);
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->active_graph) {
            for (ggml_backend_sched_kimi_async_ids_plan & plan : state->plans) {
                if (plan.state != GGML_KIMI_ASYNC_IDS_RETIRED) {
                    plan.state = GGML_KIMI_ASYNC_IDS_POISONED;
                }
            }
            for (ggml_backend_sched_kimi_async_ids_slot & slot : state->slots) {
                slot.busy = false;
            }
            state->active_graph = false;
            state->plans.clear();
            state->plan_by_split_input.clear();
        }
    }
    return !state->failed;
}

static void ggml_backend_sched_kimi_async_ids_shutdown(
        ggml_backend_sched_kimi_async_ids_state * state) {
    if (state == nullptr) return;
    (void) ggml_backend_sched_kimi_async_ids_drain(state, true);
    if (state->worker_started) {
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->stop = true;
            state->work_cv.notify_all();
        }
        state->worker.join();
        state->worker_started = false;
    }
    int slots_free = 0;
    for (const ggml_backend_sched_kimi_async_ids_slot & slot : state->slots) {
        slots_free += !slot.busy;
    }
    const bool guards_closed = !state->failed &&
        state->graphs_activated == state->graphs_completed &&
        state->plans_planned == state->plans_reserved &&
        state->plans_reserved == state->plans_submitted &&
        state->plans_submitted == state->producer_records &&
        state->producer_records == state->producer_waits &&
        state->producer_waits == state->transfer_enqueued &&
        state->transfer_enqueued == state->d2h_calls &&
        state->d2h_calls == state->done_records &&
        state->done_records == state->done_host_waits &&
        state->done_host_waits == state->plans_ready &&
        state->plans_ready == state->parse_calls &&
        state->parse_calls == state->plans_consumed &&
        state->plans_consumed == state->plans_retired &&
        state->d2h_bytes ==
            state->plans_planned * state->EXPECTED_BYTES_PER_PLAN &&
        state->tasks.empty() && state->active_tasks == 0 &&
        slots_free == state->N_SLOTS;
    GGML_LOG_INFO(
        "%s: async_expert_id_prep default_off=0 graphs_started=%llu "
        "activated=%llu declined=%llu completed=%llu poisoned=%llu "
        "planned=%llu reserved=%llu submitted=%llu producer_records=%llu "
        "producer_waits=%llu transfer_enqueued=%llu d2h_calls=%llu "
        "d2h_bytes=%llu done_records=%llu done_host_waits=%llu ready=%llu "
        "parse_calls=%llu parsed_elements=%llu consumer_waits=%llu "
        "consumer_wait_us=%llu consumer_wait_max_us=%llu consumed=%llu "
        "retired=%llu worker_wakeups=%llu queue_max=%llu slots_free=%d "
        "slot_deferrals=%llu "
        "stale_failures=%llu identity_failures=%llu layout_failures=%llu "
        "ownership_failures=%llu reorder_failures=%llu range_failures=%llu "
        "init_failures=%llu failed=%d guards_closed=%d failure_reason=%s "
        "decline_reason=%s\n",
        __func__,
        (unsigned long long) state->graphs_started,
        (unsigned long long) state->graphs_activated,
        (unsigned long long) state->graphs_declined,
        (unsigned long long) state->graphs_completed,
        (unsigned long long) state->graphs_poisoned,
        (unsigned long long) state->plans_planned,
        (unsigned long long) state->plans_reserved,
        (unsigned long long) state->plans_submitted,
        (unsigned long long) state->producer_records,
        (unsigned long long) state->producer_waits,
        (unsigned long long) state->transfer_enqueued,
        (unsigned long long) state->d2h_calls,
        (unsigned long long) state->d2h_bytes,
        (unsigned long long) state->done_records,
        (unsigned long long) state->done_host_waits,
        (unsigned long long) state->plans_ready,
        (unsigned long long) state->parse_calls,
        (unsigned long long) state->parsed_elements,
        (unsigned long long) state->consumer_waits,
        (unsigned long long) state->consumer_wait_us,
        (unsigned long long) state->consumer_wait_max_us,
        (unsigned long long) state->plans_consumed,
        (unsigned long long) state->plans_retired,
        (unsigned long long) state->worker_wakeups,
        (unsigned long long) state->queue_max,
        slots_free,
        (unsigned long long) state->slot_deferrals,
        (unsigned long long) state->stale_generation_failures,
        (unsigned long long) state->identity_failures,
        (unsigned long long) state->layout_failures,
        (unsigned long long) state->ownership_failures,
        (unsigned long long) state->reorder_failures,
        (unsigned long long) state->range_failures,
        (unsigned long long) state->init_failures,
        state->failed ? 1 : 0,
        guards_closed ? 1 : 0,
        state->failure_reason,
        state->last_decline_reason);
    ggml_backend_sched_kimi_async_ids_release_resources(state);
}

static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    struct ggml_backend_sched_split * splits = sched->splits;

    if (!ggml_backend_sched_kimi_async_d2h_begin_graph(sched)) {
        return GGML_STATUS_FAILED;
    }
    if (!ggml_backend_sched_kimi_async_h2d_begin_graph(sched)) {
        return GGML_STATUS_FAILED;
    }
    if (!ggml_backend_sched_kimi_async_ids_begin_graph(sched)) {
        return GGML_STATUS_FAILED;
    }

    ggml_tensor * prev_ids_tensor = nullptr;
    int64_t prev_ids_n_expert = -1;
    std::vector<int32_t> ids;
    std::vector<int32_t> id_counts;
    std::vector<ggml_bitset_t> used_ids;
    const bool moe_log = ggml_backend_sched_moe_log_enabled();
    const int moe_cache_slots = ggml_backend_sched_moe_cache_slots();
    const enum ggml_backend_sched_moe_prefetch_policy moe_prefetch_policy = ggml_backend_sched_moe_prefetch_policy();
    const int moe_prefetch_limit = ggml_backend_sched_moe_prefetch_limit();
    const bool split_profile = ggml_kimi_split_profile_enabled();
    const bool layer_gap_trace = g_kimi_layer_gap_trace_enabled;
    const uint64_t layer_gap_graph_compute_index = layer_gap_trace ?
        g_kimi_layer_gap_graph_compute_index.fetch_add(1, std::memory_order_relaxed) : 0;

    const auto seed_offer = ggml_cuda_moe_seed_offer_resolve();
    const auto seed_fence = ggml_cuda_moe_seed_fence_resolve();
    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];
        const int64_t split_start_us = (split_profile || layer_gap_trace) ? ggml_time_us() : 0;
        uint64_t layer_gap_moe_cache_us = 0;
        std::vector<ggml_backend_sched_moe_restore> moe_restores;
        std::vector<std::vector<uint8_t>> moe_pack_source_buffers;
        bool moe_pack_source_host_buffer_used = false;
        const bool moe_pack_source_try =
            ggml_kimi_sched_pack_source_enabled() &&
            (!ggml_kimi_sched_pack_source_prompt_only() || g_kimi_split_profile_phase == 1);

        // copy the input tensors to the split backend
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
            struct ggml_tensor * input = split->inputs[input_id];
            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

            const int async_d2h_edge = sched->kimi_async_d2h == nullptr ? -1 :
                ggml_backend_sched_kimi_async_d2h_edge_id(
                    sched->kimi_async_d2h, split_id, input_id);
            if (async_d2h_edge >= 0) {
                if (!ggml_backend_sched_kimi_async_d2h_prepare_input(
                            sched, split_id, input_id, input, input_cpy)) {
                    return GGML_STATUS_FAILED;
                }
                continue;
            }

            const int async_h2d_edge = sched->kimi_async_h2d == nullptr ? -1 :
                ggml_backend_sched_kimi_async_h2d_edge_id(
                    sched->kimi_async_h2d, split_id, input_id);
            if (async_h2d_edge >= 0) {
                if (!ggml_backend_sched_kimi_async_h2d_prepare_input(
                            sched, split_id, input_id, input, input_cpy)) {
                    return GGML_STATUS_FAILED;
                }
                continue;
            }

            if (input->flags & GGML_TENSOR_FLAG_INPUT) {
                // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }
                ggml_backend_tensor_copy(input, input_cpy);
            } else {
                const ggml_tensor * drive_home = ggml_backend_sched_drive_home(sched, input);
                const int64_t drive_wait_start_us = drive_home ? ggml_time_us() : 0;
                // wait for the split backend to finish using the input before overwriting it
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }
                if (drive_home) {
                    // a weight homed on the drive: read through the expert transport, then the next weight copy
                    // read ahead when it is homed on the drive too (an expert copy in between keeps the drive)
                    const int64_t read_start_us = ggml_time_us();
                    const auto copy_to_tensor = ggml_cuda_moe_dense_copy_to_tensor_resolve();
                    bool hit = false;
                    const bool copied = copy_to_tensor && copy_to_tensor(split_backend, input_cpy, drive_home->name, &hit);
                    const int64_t read_end_us = ggml_time_us();
                    if (!copied) {
                        ggml_backend_synchronize(input_backend);
                        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                        } else {
                            ggml_backend_synchronize(split_backend);
                        }
                        ggml_backend_tensor_copy(input, input_cpy);
                    }
                    const ggml_tensor * next = ggml_backend_sched_next_weight_input(sched, split_id, input_id);
                    const ggml_tensor * next_home = next ? ggml_backend_sched_drive_home(sched, next) : nullptr;
                    const auto read_ahead = ggml_cuda_moe_dense_read_ahead_resolve();
                    if (next_home != nullptr && read_ahead) {
                        read_ahead(next_home->name, ggml_nbytes(next_home));
                    }
                    std::lock_guard<std::mutex> lock(g_kimi_sched_pack_source_profile_mutex);
                    auto & d = g_kimi_sched_dense_stream;
                    d.tensors += 1;
                    d.bytes += ggml_nbytes(input);
                    d.read_us += (uint64_t) (read_end_us - read_start_us);
                    d.first_us += hit ? 0 : (uint64_t) (read_end_us - read_start_us);
                    d.wait_us += (uint64_t) (read_start_us - drive_wait_start_us);
                    d.readahead += hit ? 1 : 0;
                    d.fallbacks += copied ? 0 : 1;
                    continue;
                }

                // when offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used
                ggml_tensor * node = nullptr;
                if (ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(input->buffer)) {
                    for (int node_id = 0; node_id < split->graph.n_nodes; ++node_id) {
                        ggml_tensor * candidate = split->graph.nodes[node_id];
                        if (candidate->op == GGML_OP_MUL_MAT_ID && candidate->src[0] == input_cpy) {
                            node = candidate;
                            break;
                        }
                    }
                }
                if (node != nullptr) {

                    const int64_t n_expert   = input->ne[2];
                    const size_t expert_size = input->nb[2];

                    ggml_backend_synchronize(input_backend);

                    // get the ids
                    ggml_tensor * ids_tensor = node->src[2];
                    ggml_backend_t ids_backend = split_backend;

                    // if the ids tensor is also an input of the split, it may not have been copied yet to the split backend
                    // in that case, we use the original ids tensor
                    for (int i = input_id + 1; i < split->n_inputs; i++) {
                        if (ids_tensor == tensor_copy(split->inputs[i], split_backend_id, sched->cur_copy)) {
                            ids_tensor = split->inputs[i];
                            ids_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[i]);
                            break;
                        }
                    }

                    if (ids_tensor != prev_ids_tensor || n_expert != prev_ids_n_expert) {
                        const int async_ids_plan =
                            ggml_backend_sched_kimi_async_ids_plan_id(
                                sched->kimi_async_ids, split_id, input_id);
                        if (async_ids_plan >= 0) {
                            if (!ggml_backend_sched_kimi_async_ids_consume(
                                    sched, split_id, input_id, ids_tensor,
                                    ids_backend, n_expert, ids, used_ids,
                                    id_counts)) {
                                return GGML_STATUS_FAILED;
                            }
                        } else {
                            if (sched->kimi_async_ids != nullptr &&
                                    sched->kimi_async_ids->active_graph) {
                                return GGML_STATUS_FAILED;
                            }
                            ids.resize(ggml_nbytes(ids_tensor) / sizeof(int32_t));
                            ggml_backend_tensor_get_async(ids_backend, ids_tensor, ids.data(), 0, ggml_nbytes(ids_tensor));
                            ggml_backend_synchronize(ids_backend);

                            // find the used experts
                            used_ids.clear();
                            used_ids.resize(ggml_bitset_size(n_expert));
                            id_counts.clear();
                            id_counts.resize(n_expert);
                            for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; i1++) {
                                for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; i0++) {
                                    int32_t id = ids[i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t)];
                                    GGML_ASSERT(id >= 0 && id < n_expert);
                                    ggml_bitset_set(used_ids.data(), id);
                                    id_counts[id]++;
                                }
                            }
                        }

                        prev_ids_tensor = ids_tensor;
                        prev_ids_n_expert = n_expert;
                    }

                    const char * moe_cache_bypass_reason = nullptr;
                    const int64_t layer_gap_moe_cache_start_us = layer_gap_trace ? ggml_time_us() : 0;
                    const bool moe_cache_used = ggml_backend_sched_moe_cache_prepare(
                            sched, split_backend, split_backend_id, input, node, ids_tensor, ids, used_ids,
                            n_expert, expert_size, moe_cache_slots, moe_prefetch_policy, moe_prefetch_limit,
                            moe_log, &moe_cache_bypass_reason, moe_restores);
                    if (layer_gap_trace) {
                        layer_gap_moe_cache_us += (uint64_t) (ggml_time_us() - layer_gap_moe_cache_start_us);
                    }

                    if (!moe_cache_used) {
                        if (moe_log && moe_cache_slots > 0) {
                            GGML_LOG_INFO(
                                "%s: moe_cache_bypass tensor=%s node=%s ids=%s backend=%s slots=%d reason=%s n_expert=%lld expert_size=%zu\n",
                                __func__,
                                ggml_backend_sched_tensor_name(input),
                                ggml_backend_sched_tensor_name(node),
                                ggml_backend_sched_tensor_name(ids_tensor),
                                ggml_backend_name(split_backend),
                                moe_cache_slots,
                                moe_cache_bypass_reason != nullptr ? moe_cache_bypass_reason : "unknown",
                                (long long) n_expert,
                                expert_size);
                        }

                        // Group consecutive experts before selecting either the
                        // CUDA io_uring transport or the legacy host-buffer path.
                        const size_t padding = std::min<size_t>(expert_size, 512);
                        ggml_moe_pack_source::copy_plan pack_plan;
                        if (!ggml_moe_pack_source::build_copy_plan(
                                    n_expert, expert_size, padding,
                                    [&](int32_t id) {
                                        return ggml_bitset_get(used_ids.data(), id);
                                    },
                                    pack_plan)) {
                            return GGML_STATUS_FAILED;
                        }
                        const auto & range_first = pack_plan.range_first;
                        const auto & range_last = pack_plan.range_last;
                        const uint64_t expected_experts = pack_plan.expert_count;
                        const size_t expected_bytes = pack_plan.bytes;
                        size_t copy_bytes = expected_bytes;
                        int copy_ranges = (int) range_first.size();
                        bool batch_pack_source_used = false;
                        if (moe_pack_source_try && !range_first.empty() &&
                                ggml_kimi_sched_pack_source_batch_enabled()) {
                            const auto copy_ranges_to_tensor =
                                ggml_cuda_moe_expert_pack_copy_ranges_to_tensor_resolve();
                            if (copy_ranges_to_tensor) {
                                uint64_t experts_copied = 0;
                                uint64_t iouring_reads = 0;
                                uint64_t iouring_bytes = 0;
                                uint64_t cache_lookups = 0;
                                uint64_t cache_hits = 0;
                                const bool cache_bypass = ggml_kimi_sched_pack_source_cache_bypass();
                                const int64_t start_us = ggml_time_us();
                                const bool copied = copy_ranges_to_tensor(
                                    split_backend, input_cpy,
                                    ggml_backend_sched_tensor_name(input),
                                    range_first.data(), range_last.data(),
                                    range_first.size(), n_expert, expert_size,
                                    padding, cache_bypass, &experts_copied,
                                    &iouring_reads, &iouring_bytes,
                                    &cache_lookups, &cache_hits);
                                const uint64_t read_us = (uint64_t)(ggml_time_us() - start_us);
                                batch_pack_source_used = copied && experts_copied == expected_experts;
                                ggml_kimi_sched_pack_source_profile_record_batch(
                                    batch_pack_source_used, range_first.size(),
                                    expected_experts, expected_bytes, read_us,
                                    iouring_reads, iouring_bytes, cache_lookups,
                                    cache_hits, cache_bypass);
                            }
                        }

                        auto copy_experts = [&](int32_t first_id, int32_t last_id) {
                            const size_t expert_offset = first_id * expert_size;
                            const size_t expert_size_copy =  (last_id - first_id + 1) * expert_size;
                            const size_t padding_end = last_id < n_expert - 1 ? padding : 0;
                            const size_t bytes = expert_size_copy + padding_end;
                            const void * copy_src = (const uint8_t *)input->data + expert_offset;
                            size_t copy_src_bytes = bytes;

                            if (moe_pack_source_try) {
                                const void * pack_src = nullptr;
                                size_t pack_src_bytes = 0;
                                if (ggml_kimi_sched_pack_source_read_range(
                                            ggml_backend_sched_tensor_name(input),
                                            first_id,
                                            last_id,
                                            expert_size,
                                            padding_end,
                                            moe_pack_source_buffers,
                                            &pack_src,
                                            &pack_src_bytes) &&
                                        pack_src != nullptr && pack_src_bytes == bytes) {
                                    copy_src = pack_src;
                                    copy_src_bytes = pack_src_bytes;
                                    moe_pack_source_host_buffer_used = true;
                                }
                            }

                            ggml_backend_tensor_set_async(split_backend,
                                input_cpy,
                                copy_src, expert_offset,
                                // copy a bit extra at the to ensure there are no NaNs in the padding of the last expert
                                // this is necessary for MMQ in the CUDA backend
                                copy_src_bytes);

                        };

                        if (!batch_pack_source_used) {
                            for (size_t range = 0; range < range_first.size(); ++range) {
                                copy_experts(range_first[range], range_last[range]);
                            }
                        }
                        if (seed_offer) {
                            seed_offer(split_backend, input_cpy, ggml_backend_sched_tensor_name(input), n_expert,
                                expert_size, id_counts.data());
                        }

                        if (moe_log) {
                            std::string used_ids_str;
                            std::string used_id_counts_str;
                            size_t used_count = 0;
                            for (int64_t i = 0; i < n_expert; ++i) {
                                if (!ggml_bitset_get(used_ids.data(), i)) {
                                    continue;
                                }
                                if (!used_ids_str.empty()) {
                                    used_ids_str += ",";
                                }
                                used_ids_str += std::to_string(i);
                                if (!used_id_counts_str.empty()) {
                                    used_id_counts_str += ",";
                                }
                                used_id_counts_str += std::to_string(i);
                                used_id_counts_str += ":";
                                used_id_counts_str += std::to_string(id_counts[i]);
                                used_count++;
                            }

                            GGML_LOG_INFO(
                                "%s: moe_copy split=%d input=%d tensor=%s node=%s ids=%s src_backend=%s dst_backend=%s n_expert=%lld expert_size=%zu used=%zu used_bytes=%zu ranges=%d copy_bytes=%zu id_counts=[%s] ids=[%s]\n",
                                __func__,
                                split_id,
                                input_id,
                                ggml_backend_sched_tensor_name(input),
                                ggml_backend_sched_tensor_name(node),
                                ggml_backend_sched_tensor_name(ids_tensor),
                                ggml_backend_name(input_backend),
                                ggml_backend_name(split_backend),
                                (long long) n_expert,
                                expert_size,
                                used_count,
                                used_count * expert_size,
                                copy_ranges,
                                copy_bytes,
                                used_id_counts_str.c_str(),
                                used_ids_str.c_str());
                        }
                    }
                } else {
                    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
                    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
                    if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
                        ggml_backend_synchronize(input_backend);
                        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                        } else {
                            ggml_backend_synchronize(split_backend);
                        }
                        ggml_backend_tensor_copy(input, input_cpy);
                    }
                }
            }
        }

        const int64_t layer_gap_inputs_end_us = layer_gap_trace ? ggml_time_us() : 0;
        // the expert cache's seed copies out of this split's expert inputs end before the split computes: a node
        // after the expert matmul may reuse an input's memory once the matmul has read it
        if (seed_fence) {
            seed_fence(split_backend);
        }
        // The legacy path owns pageable host buffers whose lifetime ends with
        // this split, so its async copies must finish before those buffers are
        // released.  The batched path uses a process-lifetime pinned ring and
        // records a CUDA event before reusing each slot.  Its H2D copies and
        // this split's compute are on the same backend stream, so stream
        // ordering is both sufficient and necessary to retain I/O/H2D/compute
        // overlap; synchronizing here would drain the ring once per tensor.
        if (moe_pack_source_host_buffer_used) {
            const int64_t t_sync_start_us = ggml_time_us();
            ggml_backend_synchronize(split_backend);
            ggml_kimi_sched_pack_source_profile_record_sync((uint64_t) (ggml_time_us() - t_sync_start_us));
        }

        auto restore_moe_cache_nodes = [&]() {
            for (ggml_backend_sched_moe_restore & restore : moe_restores) {
                restore.node->src[0] = restore.src0;
                restore.node->src[2] = restore.src2;
            }
        };

        if (!ggml_backend_sched_kimi_async_d2h_consumer_begin(
                    sched, split_id) ||
                !ggml_backend_sched_kimi_async_h2d_consumer_begin(
                    sched, split_id)) {
            restore_moe_cache_nodes();
            return GGML_STATUS_FAILED;
        }

        const int64_t layer_gap_compute_start_us = layer_gap_trace ? ggml_time_us() : 0;
        if (!sched->callback_eval) {
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            if (ec != GGML_STATUS_SUCCESS) {
                restore_moe_cache_nodes();
                return ec;
            }
        } else {
            // similar to ggml_backend_compare_graph_backend
            for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
                struct ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

                int j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_data);
                }

                struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                if (ec != GGML_STATUS_SUCCESS) {
                    restore_moe_cache_nodes();
                    return ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);

                if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                    break;
                }

                j0 = j1;
            }
        }
        if (!ggml_backend_sched_kimi_async_d2h_producer_submitted(
                    sched, split_id) ||
                !ggml_backend_sched_kimi_async_h2d_producer_submitted(
                    sched, split_id) ||
                !ggml_backend_sched_kimi_async_ids_producer_submitted(
                    sched, split_id) ||
                !ggml_backend_sched_kimi_async_d2h_consumer_end(
                    sched, split_id) ||
                !ggml_backend_sched_kimi_async_h2d_consumer_end(
                    sched, split_id)) {
            restore_moe_cache_nodes();
            return GGML_STATUS_FAILED;
        }
        const int64_t layer_gap_compute_end_us = layer_gap_trace ? ggml_time_us() : 0;
        restore_moe_cache_nodes();
        // record the event of this copy
        if (split->n_inputs > 0) {
            if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
            }
        }
        const int64_t layer_gap_event_end_us = layer_gap_trace ? ggml_time_us() : 0;
        if (split_profile) {
            ggml_kimi_split_profile_record(split, split_backend, (uint64_t) (ggml_time_us() - split_start_us));
        }
        ggml_kimi_split_moe_assign_profile_record(sched, split_id, split, split_backend);
        if (layer_gap_trace) {
            ggml_kimi_layer_gap_sched_entry entry;
            entry.graph_compute_index = layer_gap_graph_compute_index;
            entry.phase = g_kimi_split_profile_phase;
            entry.split_id = split_id;
            entry.n_nodes = split->graph.n_nodes;
            entry.split_start_us = (uint64_t) split_start_us;
            entry.inputs_end_us = (uint64_t) layer_gap_inputs_end_us;
            entry.compute_start_us = (uint64_t) layer_gap_compute_start_us;
            entry.compute_end_us = (uint64_t) layer_gap_compute_end_us;
            entry.event_end_us = (uint64_t) layer_gap_event_end_us;
            entry.split_end_us = (uint64_t) ggml_time_us();
            entry.moe_cache_us = layer_gap_moe_cache_us;
            ggml_kimi_layer_gap_copy_field(
                entry.backend, sizeof(entry.backend), ggml_backend_name(split_backend));
            if (split->graph.n_nodes > 0) {
                ggml_kimi_layer_gap_copy_field(
                    entry.first, sizeof(entry.first), ggml_backend_sched_tensor_name(split->graph.nodes[0]));
                ggml_kimi_layer_gap_copy_field(
                    entry.last,
                    sizeof(entry.last),
                    ggml_backend_sched_tensor_name(split->graph.nodes[split->graph.n_nodes - 1]));
            }
            ggml_kimi_layer_gap_sched_thread_buffer().entries.push_back(entry);
        }
    }

    if (seed_fence) { // a split that offered and never computed
        for (int b = 0; b < sched->n_backends; b++) {
            seed_fence(sched->backends[b]);
        }
    }
    return ggml_backend_sched_kimi_async_d2h_finish_graph(sched) &&
        ggml_backend_sched_kimi_async_h2d_finish_graph(sched) &&
        ggml_backend_sched_kimi_async_ids_finish_graph(sched) ?
        GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));

    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    const char * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (int *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
    sched->debug_causes = sched->debug > 1 ?
        (char (*)[64]) calloc(sched->hash_set.size, sizeof(sched->debug_causes[0])) : nullptr;

    const size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

    sched->debug_graph_size = 0;
    sched->debug_prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
    sched->context_buffer = (char *) malloc(sched->context_buffer_size);

    const int initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, sizeof(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    for (int b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        if (sched->n_copies > 1) {
            for (int c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;
    sched->drive_homes = new std::unordered_map<const struct ggml_tensor *, bool>();
    sched->moe_caches = new std::vector<ggml_backend_sched_moe_cache *>();
    sched->decode_phase = true;
    sched->kimi_async_d2h =
        ggml_backend_sched_kimi_env_enabled(
            "GGML_KIMI_ASYNC_SPLIT_INPUT_D2H") ?
        new ggml_backend_sched_kimi_async_d2h_state() : nullptr;
    sched->kimi_async_h2d =
        ggml_backend_sched_kimi_env_enabled(
            "GGML_KIMI_ASYNC_SPLIT_INPUT_H2D") ?
        new ggml_backend_sched_kimi_async_h2d_state() : nullptr;
    sched->kimi_async_ids =
        ggml_backend_sched_kimi_env_enabled(
            "GGML_KIMI_ASYNC_EXPERT_ID_PREP") ?
        new ggml_backend_sched_kimi_async_ids_state() : nullptr;

    ggml_backend_sched_reset(sched);

    return sched;
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    ggml_backend_sched_kimi_async_d2h_shutdown(sched->kimi_async_d2h);
    delete sched->kimi_async_d2h;
    sched->kimi_async_d2h = nullptr;
    ggml_backend_sched_kimi_async_h2d_shutdown(sched->kimi_async_h2d);
    delete sched->kimi_async_h2d;
    sched->kimi_async_h2d = nullptr;
    ggml_backend_sched_kimi_async_ids_shutdown(sched->kimi_async_ids);
    delete sched->kimi_async_ids;
    sched->kimi_async_ids = nullptr;
    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    for (ggml_backend_sched_moe_cache * cache : *sched->moe_caches) {
        ggml_backend_buffer_free(cache->weights_buffer);
        ggml_backend_buffer_free(cache->ids_buffer);
        delete cache;
    }
    delete sched->moe_caches;
    delete sched->drive_homes;
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    free(sched->splits);
    free(sched->hv_tensor_backend_ids);
    free(sched->hv_tensor_copies);
    free(sched->debug_causes);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
    free(sched);
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    if (sched->kimi_async_d2h != nullptr &&
            sched->kimi_async_d2h->active_graph) {
        GGML_ASSERT(ggml_backend_sched_kimi_async_d2h_drain(
            sched->kimi_async_d2h, true));
    }
    if (sched->kimi_async_h2d != nullptr &&
            sched->kimi_async_h2d->active_graph) {
        GGML_ASSERT(ggml_backend_sched_kimi_async_h2d_drain(
            sched->kimi_async_h2d, true));
    }
    if (sched->kimi_async_ids != nullptr &&
            sched->kimi_async_ids->active_graph) {
        GGML_ASSERT(ggml_backend_sched_kimi_async_ids_drain(
            sched->kimi_async_ids, true));
    }
    // reset state for the next run
    if (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
        if (sched->debug_causes != nullptr) {
            memset(sched->debug_causes, 0, sched->hash_set.size * sizeof(sched->debug_causes[0]));
        }
        sched->is_reset = true;
    }
    sched->is_alloc = false;
}

void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }

    // each backend sizes its scratch memory for its splits now, rather than while the graph runs
    bool ok = true;
    std::vector<struct ggml_cgraph *> graphs;
    for (int b = 0; b < sched->n_backends && ok; b++) {
        ggml_backend_t backend = sched->backends[b];
        if (backend->iface.graph_reserve == NULL) {
            continue;
        }
        graphs.clear();
        for (int i = 0; i < sched->n_splits; i++) {
            if (sched->splits[i].backend_id == b) {
                graphs.push_back(&sched->splits[i].graph);
            }
        }
        ok = backend->iface.graph_reserve(backend, graphs.data(), (int) graphs.size());
    }

    ggml_backend_sched_reset(sched);

    return ok;
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }

    sched->is_alloc = true;

    return true;
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    return err;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    const enum ggml_status status = ggml_backend_sched_compute_splits(sched);
    if (status != GGML_STATUS_SUCCESS && sched->kimi_async_d2h != nullptr) {
        (void) ggml_backend_sched_kimi_async_d2h_drain(
            sched->kimi_async_d2h, true);
    }
    if (status != GGML_STATUS_SUCCESS && sched->kimi_async_h2d != nullptr) {
        (void) ggml_backend_sched_kimi_async_h2d_drain(
            sched->kimi_async_h2d, true);
    }
    if (status != GGML_STATUS_SUCCESS && sched->kimi_async_ids != nullptr) {
        (void) ggml_backend_sched_kimi_async_ids_drain(
            sched->kimi_async_ids, true);
    }
    return status;
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    if (sched->kimi_async_d2h != nullptr &&
            sched->kimi_async_d2h->active_graph) {
        (void) ggml_backend_sched_kimi_async_d2h_drain(
            sched->kimi_async_d2h, true);
    }
    if (sched->kimi_async_h2d != nullptr &&
            sched->kimi_async_h2d->active_graph) {
        (void) ggml_backend_sched_kimi_async_h2d_drain(
            sched->kimi_async_h2d, true);
    }
    if (sched->kimi_async_ids != nullptr &&
            sched->kimi_async_ids->active_graph) {
        (void) ggml_backend_sched_kimi_async_ids_drain(
            sched->kimi_async_ids, true);
    }
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    if (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_splits;
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_copies;
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    return sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");
    sched->is_reset = false;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    GGML_ASSERT(sched);
    int backend_index = tensor_backend_id(node);
    if (backend_index == -1) {
        return NULL;
    }
    return sched->backends[backend_index];
}

// utils

enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    return ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (char *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (char *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct ggml_tensor * graph_copy_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,
    struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) {

    GGML_ASSERT(src != NULL);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(&hash_set, src);
    if (id == GGML_HASHSET_ALREADY_EXISTS) {
        return node_copies[ggml_hash_find(&hash_set, src)];
    }

    struct ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_copy_init_tensor(struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        enum ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        graph_copy_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    struct ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    struct ggml_tensor ** node_copies = (ggml_tensor **) calloc(hash_set.size, sizeof(node_copies[0])); // NOLINT
    bool * node_init = (bool *) calloc(hash_set.size, sizeof(node_init[0]));

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    free(node_copies);
    free(node_init);

    return {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    if (num_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        bool verified = false;
        for (int i = 0; i < g1->n_nodes; i++) {
            for (size_t j = 0; j < num_test_nodes; ++j) {
                if (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = true;
                }
            }
        }
        GGML_ASSERT(verified);
    } else {
        for (int i = 0; i < g1->n_nodes; i++) {
            struct ggml_tensor * t1 = g1->nodes[i];
            struct ggml_tensor * t2 = g2->nodes[i];

            assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            if (ggml_is_view_op(t1->op)) {
                continue;
            }

            // compare results, calculate rms etc
            if (!callback(i, t1, t2, user_data)) {
                break;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    return true;
}

// CPU backend - buffer

static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

static void ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

static const char * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}
