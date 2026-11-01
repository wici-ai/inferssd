// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stddef.h>
#include <stdint.h>

// Borrowed descriptors must remain open until run returns. No tensor payload is
// accessed: resolve the packed expert by its tensor name and expert index.
struct kimi_cpu_sink_source {
    int fd_direct;
    uint64_t offset;
    size_t nbytes;
    size_t data_offset;
    size_t payload_nbytes;
};
