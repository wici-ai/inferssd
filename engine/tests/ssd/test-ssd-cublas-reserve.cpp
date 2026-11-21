// SPDX-License-Identifier: Apache-2.0
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"

#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

static unsigned creates;
static unsigned reclaims;
static bool fail_create;

extern "C" cublasStatus_t cublasCreate(cublasHandle_t * handle) {
    ++creates;
    if (fail_create) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    static auto real_create = reinterpret_cast<decltype(&cublasCreate)>(dlsym(RTLD_NEXT, "cublasCreate_v2"));
    GGML_ASSERT(real_create);
    return real_create(handle);
}

static int cannot_reclaim(void *, size_t requested, size_t * freed) {
    GGML_ASSERT(requested == 0);
    ++reclaims;
    *freed = 0;
    return 0;
}

int main() {
    if (ggml_backend_cuda_get_device_count() < 1) {
        std::fprintf(stderr, "This fixture requires a visible NVIDIA GPU.\n");
        return EXIT_FAILURE;
    }
    auto backend = ggml_backend_cuda_init(0);
    GGML_ASSERT(backend);
    GGML_ASSERT(ggml_cuda_oom_reclaimer_register(0, cannot_reclaim, &reclaims));
    auto hold = reinterpret_cast<void (*)(ggml_backend_t, bool)>(
            ggml_backend_reg_get_proc_address(ggml_backend_cuda_reg(), "ggml_backend_pool_hold"));
    GGML_ASSERT(hold);
    hold(backend, true);

    auto ctx = ggml_init({1024 * 1024, nullptr, true});
    GGML_ASSERT(ctx);
    auto vector_graph = ggml_new_graph_custom(ctx, 32, false);
    auto matrix_graph = ggml_new_graph_custom(ctx, 32, false);
    auto weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 256);
    auto vector = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1);
    ggml_build_forward_expand(vector_graph, ggml_mul_mat(ctx, weights, vector));
    // An unaligned inner dimension excludes the custom matrix kernel.
    auto left = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 257, 256);
    auto right = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 257, 256);
    ggml_build_forward_expand(matrix_graph, ggml_mul_mat(ctx, left, right));
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);
    ggml_backend_buffer_clear(buffer, 0);

    const auto initial = creates;
    GGML_ASSERT(backend->iface.graph_reserve(backend, &vector_graph, 1));
    GGML_ASSERT(ggml_backend_graph_compute(backend, vector_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    GGML_ASSERT(creates == initial);

    fail_create = true;
    GGML_ASSERT(!backend->iface.graph_reserve(backend, &matrix_graph, 1));
    GGML_ASSERT(creates == initial + 1 && reclaims == 1);
    fail_create = false;
    GGML_ASSERT(backend->iface.graph_reserve(backend, &matrix_graph, 1));
    GGML_ASSERT(creates == initial + 2);
    GGML_ASSERT(backend->iface.graph_reserve(backend, &matrix_graph, 1));
    GGML_ASSERT(ggml_backend_graph_compute(backend, matrix_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    GGML_ASSERT(creates == initial + 2 && reclaims == 1);

    GGML_ASSERT(ggml_cuda_oom_reclaimer_unregister(0, cannot_reclaim, &reclaims));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    std::puts("PASS custom matvec, cuBLAS reserve failure, retry and reuse");
}
