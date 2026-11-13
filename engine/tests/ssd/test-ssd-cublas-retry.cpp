// SPDX-License-Identifier: Apache-2.0
#include <cublas_v2.h>
#include "ggml-cuda.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

// Link to the backend's real retry helper. Interpose only cublasCreate_v2 in this
// test executable so failure states are deterministic without exhausting VRAM.
cublasStatus_t ggml_cuda_cublas_create_reclaiming(cublasHandle_t *, int);
static std::vector<cublasStatus_t> sequence;
static size_t calls, reclaim_calls;
static int callback_result;
static size_t callback_freed;
static bool bad_request;
extern "C" cublasStatus_t CUBLASWINAPI cublasCreate_v2(cublasHandle_t * handle) {
    if (calls >= sequence.size()) std::abort();
    const auto status = sequence[calls++];
    if (status == CUBLAS_STATUS_SUCCESS) *handle = reinterpret_cast<cublasHandle_t>(uintptr_t(1));
    return status;
}
static int reclaim(void *, size_t requested, size_t * freed) {
    ++reclaim_calls;
    if (requested != 0) bad_request = true; // opaque allocation size is unknown
    *freed = callback_freed;
    return callback_result;
}
static void check(const char * name, std::vector<cublasStatus_t> statuses, bool registered,
                  int reclaimed, size_t freed, cublasStatus_t expected, size_t attempts, size_t expected_reclaims) {
    sequence = statuses; calls = reclaim_calls = 0; bad_request = false;
    callback_result = reclaimed; callback_freed = freed;
    if (registered && !ggml_cuda_oom_reclaimer_register(0, reclaim, &calls)) std::abort();
    cublasHandle_t handle = reinterpret_cast<cublasHandle_t>(uintptr_t(2));
    const auto got = ggml_cuda_cublas_create_reclaiming(&handle, 0);
    if (registered && !ggml_cuda_oom_reclaimer_unregister(0, reclaim, &calls)) std::abort();
    if (got != expected || calls != attempts || reclaim_calls != expected_reclaims || bad_request ||
        handle != (got == CUBLAS_STATUS_SUCCESS ? reinterpret_cast<cublasHandle_t>(uintptr_t(1)) : nullptr)) {
        std::fprintf(stderr, "FAIL %s status=%d calls=%zu reclaims=%zu\n", name, int(got), calls, reclaim_calls);
        std::exit(1);
    }
    std::printf("PASS %s\n", name);
}
int main() {
    check("success preserves handle", {CUBLAS_STATUS_SUCCESS}, true, 1, 4096, CUBLAS_STATUS_SUCCESS, 1, 0);
    check("allocation failure retries after progress", {CUBLAS_STATUS_ALLOC_FAILED,CUBLAS_STATUS_ALLOC_FAILED,CUBLAS_STATUS_SUCCESS}, true, 1, 4096, CUBLAS_STATUS_SUCCESS, 3, 2);
    check("no reclaimer", {CUBLAS_STATUS_ALLOC_FAILED}, false, 0, 0, CUBLAS_STATUS_ALLOC_FAILED, 1, 0);
    check("busy or disarmed reclaimer", {CUBLAS_STATUS_ALLOC_FAILED}, true, 0, 0, CUBLAS_STATUS_ALLOC_FAILED, 1, 1);
    check("deferred reclaim does not spin", {CUBLAS_STATUS_ALLOC_FAILED}, true, GGML_CUDA_RECLAIM_LATER, 0, CUBLAS_STATUS_ALLOC_FAILED, 1, 1);
    check("zero progress terminates", {CUBLAS_STATUS_ALLOC_FAILED}, true, 1, 0, CUBLAS_STATUS_ALLOC_FAILED, 1, 1);
    check("callback failure terminates", {CUBLAS_STATUS_ALLOC_FAILED}, true, -1, 0, CUBLAS_STATUS_ALLOC_FAILED, 1, 1);
    check("other status not retried", {CUBLAS_STATUS_NOT_INITIALIZED}, true, 1, 4096, CUBLAS_STATUS_NOT_INITIALIZED, 1, 0);
}
