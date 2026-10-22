#include "ggml.h"

#include <cstdio>
#include <initializer_list>

int main() {
    const size_t sizes[] = {1, 2, 3, 31, 64, 257, 2048, 8192};
    for (const size_t size : sizes) {
        for (const bool grads : {false, true}) {
            const size_t bytes = ggml_graph_overhead_custom(size, grads);
            const ggml_init_params params = {bytes, nullptr, true};
            ggml_context * ctx = ggml_init(params);
            GGML_ASSERT(ctx != nullptr);
            // Exactly the advertised allocation must hold every graph array,
            // including gradient arrays and hash/alignment boundary cases.
            GGML_ASSERT(ggml_new_graph_custom(ctx, size, grads) != nullptr);
            GGML_ASSERT(ggml_used_mem(ctx) == bytes);
            ggml_free(ctx);
        }
    }
    std::puts("graph sizing and allocation agree for 16 layouts");
    return 0;
}
