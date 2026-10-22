#pragma once
#include "kimi-cache-location-api.h"

// View of an existing kimi_vram_tier context. This does not create a second
// cache. The CPU owner supplies geometry before acquire_native; all callbacks
// remain serialized by its run_mutex. The DSO must outlive that owner.
struct kimi_vram_location_api {
    uint32_t version; // 1
    uint32_t size;
    const kimi_cache_location_api * view;
    int (*configure)(void *, int latent, int hidden, char *, size_t);
};
using kimi_vram_location_get_api_fn = const kimi_vram_location_api * (*)();
