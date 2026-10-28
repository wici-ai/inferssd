// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ggml-backend.h"
#include <stddef.h>
struct ggml_ssd_mac_cache;
#ifdef __cplusplus
namespace ssd {
namespace settings { class snapshot; }
struct runtime_services;
namespace mac {
// Set/detach only while the backend is quiescent. Registry private proc.
using set_backend_cache = bool (*)(ggml_backend_t, void *) noexcept;
// The caller owns the cache and joins graph callers before destroying it.
// Workers and pending reads join before destroy returns; services are borrowed.
GGML_BACKEND_API ggml_ssd_mac_cache * create_configured(const settings::snapshot *, const runtime_services *, char *, size_t) noexcept;
GGML_BACKEND_API void destroy(ggml_ssd_mac_cache *) noexcept;
} }
#endif
