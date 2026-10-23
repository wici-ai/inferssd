// SPDX-License-Identifier: MIT
#include "ggml-cuda/ssd/reserve-registry.h"
#include "ggml-ssd/runtime-services.h"

#include <cstdio>
#include <cstdlib>

static void require(bool condition) {
    if (!condition) { std::abort(); }
}

int main() {
    ssd::gpu::reserve_registry<2> registry;
    const int first = 1, second = 2, other = 3, zero = 4;
    constexpr size_t mib = size_t{1} << 20;
    require(registry.maximum(0) == 0 && registry.maximum(1) == 0);
    require(registry.add(&first, 0, 1024 * mib) == ssd::status::ok);
    require(registry.add(&second, 0, 2048 * mib) == ssd::status::ok);
    require(registry.maximum(0) == 2048 * mib);
    require(registry.add(&other, 1, 4096 * mib) == ssd::status::ok);
    require(registry.maximum(0) == 2048 * mib && registry.maximum(1) == 4096 * mib);
    // Idempotent attach must not replace an owner's immutable registration.
    require(registry.add(&first, 0, 1024 * mib) == ssd::status::ok);
    require(registry.add(&first, 0, 3072 * mib) == ssd::status::invalid);
    require(registry.add(&first, 1, 1024 * mib) == ssd::status::invalid);
    require(registry.maximum(0) == 2048 * mib && registry.maximum(1) == 4096 * mib);
    require(registry.remove(&second) == ssd::status::ok && registry.maximum(0) == 1024 * mib);
    require(registry.remove(&second) == ssd::status::ok && registry.maximum(0) == 1024 * mib);
    require(registry.add(&zero, 0, 0) == ssd::status::ok);
    require(registry.remove(&first) == ssd::status::ok && registry.maximum(0) == 0);
    require(registry.remove(&zero) == ssd::status::ok && registry.maximum(0) == 0);
    require(registry.remove(&other) == ssd::status::ok && registry.maximum(1) == 0);
    require(registry.add(nullptr, 0, 1) == ssd::status::invalid);
    require(registry.add(&first, -1, 1) == ssd::status::invalid);
    require(registry.add(&first, 2, 1) == ssd::status::invalid);
    require(registry.remove(nullptr) == ssd::status::invalid);
    require(registry.maximum(-1) == 0 && registry.maximum(2) == 0);
    const ssd::runtime_services services{};
    require(services.gpu_reserve_bytes == 0);
    std::puts("SSD_CUDA_CONTEXT_RESERVE_PASS");
}
