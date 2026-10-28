// Legacy process-global publication contract, test reference only.
#pragma once
#include "ggml-ssd/store/catalog.h"
namespace c0 {
// The one instance is defined in a shared library, never as an inline singleton.
// Providers may populate it only inside their registered initializer. Once
// published, vectors and file descriptors must remain stable until shutdown.
parser_state & shared_catalog();
struct provider {
    void (*initialize)();
    const expert_pack_entry * (*lookup)(const char *, int, size_t);
};
bool register_provider(const provider * value);
bool unregister_provider(const provider * value);
bool initialize_shared_catalog();
// Does not initialize. Diagnostics must not inspect vectors until publication.
// A retired provider makes the catalog unavailable even while its fds live.
bool catalog_available();
} // namespace c0
