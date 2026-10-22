#include "moe-adapter-dispatch.h"

#include <atomic>
#include <cstdio>

namespace ggml_moe {

enum adapter_kind : int {
    adapter_unselected = 0,
    adapter_common = 1,
    adapter_kimi_k3_local = 2,
};

static std::atomic<int> g_adapter{adapter_unselected};
static std::atomic<bool> g_adapter_mismatch_reported{false};

// Kimi-K3's local cache and transport are validated for these Q2_K and IQ1_S
// role-slice sizes. Matching only the model family or expert count is
// insufficient: another 896-expert layout must remain on the common adapter.
static constexpr int64_t KIMI_K3_LOCAL_EXPERTS = 896;
static constexpr size_t KIMI_K3_Q2_K_ROLE_BYTES = 3612672;
static constexpr size_t KIMI_K3_IQ1_S_ROLE_BYTES = 2150400;

void adapter_select_model_shape(
        int64_t n_experts,
        size_t expert_bytes,
        const char * source) {
    if (n_experts <= 0 || expert_bytes == 0) {
        return;
    }
    const bool supported_kimi_slice =
        expert_bytes == KIMI_K3_Q2_K_ROLE_BYTES ||
        expert_bytes == KIMI_K3_IQ1_S_ROLE_BYTES;
    const adapter_kind selected =
        n_experts == KIMI_K3_LOCAL_EXPERTS && supported_kimi_slice
            ? adapter_kimi_k3_local
            : adapter_common;

    int expected = adapter_unselected;
    if (g_adapter.compare_exchange_strong(
            expected, selected, std::memory_order_acq_rel, std::memory_order_acquire)) {
        std::fprintf(stderr,
            "[moe_adapter] dispatch=%s source=%s n_experts=%lld "
            "expert_bytes=%zu selection=%s\n",
            selected == adapter_common ? "common" : "k3-local",
            source ? source : "unknown",
            (long long)n_experts,
            expert_bytes,
            "model-shape");
        return;
    }

    if (expected != selected &&
            !g_adapter_mismatch_reported.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr,
            "[moe_adapter] shape mismatch ignored: active=%s requested=%s "
            "source=%s n_experts=%lld expert_bytes=%zu\n",
            expected == adapter_common ? "common" : "k3-local",
            selected == adapter_common ? "common" : "k3-local",
            source ? source : "unknown",
            (long long)n_experts,
            expert_bytes);
    }
}

bool adapter_uses_common() {
    // Unknown shape is deliberately conservative.  The CPU bridge selects
    // before normal cache probes; if an earlier probe does occur, it must not
    // activate K3-local without a validated shape.
    return g_adapter.load(std::memory_order_acquire) != adapter_kimi_k3_local;
}

bool adapter_is_selected() {
    return g_adapter.load(std::memory_order_acquire) != adapter_unselected;
}

} // namespace ggml_moe

extern "C" void ggml_cuda_moe_adapter_select_model_shape(
        int64_t n_experts,
        size_t expert_bytes) {
    ggml_moe::adapter_select_model_shape(n_experts, expert_bytes, "cpu-bridge");
}
