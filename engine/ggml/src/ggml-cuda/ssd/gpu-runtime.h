// SPDX-License-Identifier: Apache-2.0
#pragma once
// The GPU runtime of the backend the tier is built into, under the CUDA runtime's names: CUDA's own, or HIP's through
// the backend's vendor header (vendors/hip.h), which maps those names for every source of the backend (common.cuh).
// A source that includes this is compiled in the backend's language (CUDA, or HIP: ssd/CMakeLists.txt).
#if defined(GGML_USE_HIP)
#include "../vendors/hip.h"
#else
#include <cuda_runtime.h>
#endif
