// SPDX-License-Identifier: Apache-2.0
#undef NDEBUG
#include "ggml-ssd/gpu/provider.h"
#include <cassert>
#include <stdexcept>

namespace {
void quantize(const float *, void *, int64_t) {}
void reference(int, float *, size_t, const void *, size_t, const void *, size_t, int) {}
int input_type(int) { return GGML_TYPE_Q8_0; }
int invalid_type(int) { return GGML_TYPE_COUNT; }
int throwing_type(int) { throw std::runtime_error("injected provider failure"); }
}
int main() {
    const ssd::gpu::cpu_kernels_v1 good{1, sizeof(ssd::gpu::cpu_kernels_v1), quantize, reference, input_type};
    assert(ssd::gpu::valid(&good));
    assert(!ssd::gpu::valid(static_cast<const ssd::gpu::cpu_kernels_v1 *>(nullptr)));
    auto value = good;
    value.version = 2; assert(!ssd::gpu::valid(&value));
    value = good; --value.size; assert(!ssd::gpu::valid(&value));
    value = good; value.quantize_q8_0 = nullptr; assert(!ssd::gpu::valid(&value));
    value = good; value.reference_mxfp4 = nullptr; assert(!ssd::gpu::valid(&value));
    value = good; value.vec_dot_type = nullptr; assert(!ssd::gpu::valid(&value));
    value = good; value.vec_dot_type = invalid_type; assert(!ssd::gpu::valid(&value));
    value = good; value.vec_dot_type = throwing_type; assert(!ssd::gpu::valid(&value));
    ssd::platform::file_identity file;
    file.device = 7; file.id_lo = 11; file.id_hi = 29; file.size = 4096;
    file.mtime_ns = 13000000017; file.ctime_ns = -1000000000 + 23; // before the epoch: -1 s + 23 ns
    const auto source = kimi_vram_source_identity(file, 32, 64);
    assert(source.device == 7 && source.inode == 11 && source.inode_hi == 29 && source.size == 4096 &&
           source.offset == 32 && source.bytes == 64);
    // Seconds and nanoseconds as POSIX stat splits them.
    assert(source.mtime_s == 13 && source.mtime_ns == 17 && source.ctime_s == -1 && source.ctime_ns == 23);
    file.ctime_ns = -1; // 1 ns before the epoch
    const auto early = kimi_vram_source_identity(file, 32, 64);
    assert(early.ctime_s == -1 && early.ctime_ns == 999999999);
}
