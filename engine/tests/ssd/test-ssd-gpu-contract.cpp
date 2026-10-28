// SPDX-License-Identifier: Apache-2.0
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
    struct stat st{};
    st.st_dev = 7; st.st_ino = 11; st.st_size = 4096;
#if defined(__APPLE__)
    st.st_mtimespec = {13, 17}; st.st_ctimespec = {19, 23};
#elif defined(_WIN32)
    st.st_mtime = 13; st.st_ctime = 19;
#else
    st.st_mtim = {13, 17}; st.st_ctim = {19, 23};
#endif
    const auto source = kimi_vram_source_identity(&st, 32, 64);
    assert(source.device == 7 && source.inode == 11 && source.size == 4096 && source.offset == 32 && source.bytes == 64);
    assert(source.mtime_s == 13 && source.ctime_s == 19);
#if !defined(_WIN32)
    assert(source.mtime_ns == 17 && source.ctime_ns == 23);
#endif
}
