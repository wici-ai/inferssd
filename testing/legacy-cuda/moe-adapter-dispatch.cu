#include "moe-adapter-dispatch.h"

#include <atomic>
#include <cstdio>

#if defined(GGML_SSD_STORE)
#include "store.h"
namespace ggml_moe {
void adapter_select_model_shape(int64_t experts, size_t bytes, const char * source) {
    ggml_ssd_c0_select_model_shape(experts, bytes, source);
}
bool adapter_uses_common() { return ggml_ssd_c0_uses_common(); }
bool adapter_is_selected() { return ggml_ssd_c0_shape_selected(); }
} // namespace ggml_moe
#else
namespace ggml_moe {

enum adapter_kind : int {
    adapter_unselected = 0,
    adapter_common = 1,
};

static std::atomic<int> g_adapter{adapter_unselected};

void adapter_select_model_shape(
        int64_t n_experts,
        size_t expert_bytes,
        const char * source) {
    if (n_experts <= 0 || expert_bytes == 0) {
        return;
    }
    int expected = adapter_unselected;
    if (g_adapter.compare_exchange_strong(
            expected, adapter_common, std::memory_order_acq_rel, std::memory_order_acquire)) {
        std::fprintf(stderr,
            "[moe_adapter] dispatch=%s source=%s n_experts=%lld "
            "expert_bytes=%zu selection=%s\n",
            "common",
            source ? source : "unknown",
            (long long)n_experts,
            expert_bytes,
            "model-shape");
        return;
    }

}

bool adapter_uses_common() { return true; }

bool adapter_is_selected() {
    return g_adapter.load(std::memory_order_acquire) != adapter_unselected;
}

} // namespace ggml_moe

#endif

extern "C" void ggml_cuda_moe_adapter_select_model_shape(
        int64_t n_experts,
        size_t expert_bytes) {
    ggml_moe::adapter_select_model_shape(n_experts, expert_bytes, "cpu-bridge");
}
