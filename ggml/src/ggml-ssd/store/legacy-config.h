#pragma once
#include "legacy-index.h"

namespace c0 {

// Match mode/sidecar precedence only to refuse active unsupported transport.
// Explicit 0/off/disabled remains inert even when a sidecar path is present.
inline bool lossless_requested() {
    const char * path = std::getenv("GGML_MOE_LOSSLESS_PACK");
    if (!path || !path[0]) { path = std::getenv("GGML_MOE_GLM52_LOSSLESS_PACK"); }
    const char * mode = std::getenv("GGML_MOE_LOSSLESS_MODE");
    if (!mode || !mode[0]) { mode = std::getenv("GGML_MOE_GLM52_LOSSLESS_MODE"); }
    if (!mode || !mode[0]) { return path && path[0]; }
    return std::strcmp(mode, "0") != 0 && std::strcmp(mode, "off") != 0 && std::strcmp(mode, "disabled") != 0;
}

// Keep the legacy initializer's order and distinction between a single path
// and a colon-separated *_LIST. In particular, overlays require a base PACK.
inline description description_from_env() {
    description desc = source_description_from_env();
    const char * io = std::getenv("GGML_MOE_IO_BACKEND");
    if (io && io[0]) {
        if (std::strcmp(io, "direct") == 0) {
            desc.io_backend = 1;
        } else if (std::strcmp(io, "iouring") == 0 || std::strcmp(io, "io_uring") == 0) {
            desc.io_backend = 2;
        } else if (std::strcmp(io, "mmap") != 0) {
            std::fprintf(stderr,
                "[moe_stream_batch] GGML_MOE_IO_BACKEND=%s requested; runtime supports mmap/buffered, direct, and iouring expert-pack reads\n", io);
        }
    }

    return desc;
}
} // namespace c0
