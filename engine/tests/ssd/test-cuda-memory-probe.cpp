// SPDX-License-Identifier: Apache-2.0
#include "ggml-cuda.h"
#include <cuda_runtime_api.h>
#include <cstdio>
#include <dlfcn.h>

static bool busy;

extern "C" cudaError_t cudaMemGetInfo(size_t * free, size_t * total) {
    if (busy) {
        return cudaErrorMemoryAllocation;
    }
    static auto query = reinterpret_cast<decltype(&cudaMemGetInfo)>(dlsym(RTLD_NEXT, "cudaMemGetInfo"));
    GGML_ASSERT(query);
    return query(free, total);
}

int main() {
    if (!ggml_backend_cuda_get_device_count()) {
        std::fprintf(stderr, "This fixture requires a visible NVIDIA GPU.\n");
        return 77;
    }
    size_t free = 1, total = 1;
    busy = true;
    ggml_backend_cuda_get_device_memory(0, &free, &total);
    GGML_ASSERT(free == 0 && total == 0);
    busy = false;
    ggml_backend_cuda_get_device_memory(0, &free, &total);
    GGML_ASSERT(total > 0 && free <= total);
    return 0;
}
