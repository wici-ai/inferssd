#include "activation-quant-reuse.cuh"

#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

static bool activation_quant_reuse_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("GGML_CUDA_DENSE_ACTIVATION_QUANT_REUSE");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0 &&
            std::strcmp(value, "false") != 0;
    }();
    return enabled;
}

struct activation_quant_reuse_entry {
    const ggml_tensor * src = nullptr;
    const void * src_data = nullptr;
    ggml_cuda_activation_quant_type type = GGML_CUDA_ACTIVATION_QUANT_Q8_1;
    void * data = nullptr;
    ggml_cuda_pool * pool = nullptr;
    size_t size = 0;
    size_t actual_size = 0;
};

struct activation_quant_reuse_state {
    ggml_backend_cuda_context * ctx = nullptr;
    std::vector<activation_quant_reuse_entry> entries;
    size_t depth = 0;
};

static thread_local std::vector<activation_quant_reuse_state> activation_quant_reuse_states;

static activation_quant_reuse_state * activation_quant_reuse_find(
        ggml_backend_cuda_context & ctx) {
    for (activation_quant_reuse_state & state : activation_quant_reuse_states) {
        if (state.ctx == &ctx) {
            return &state;
        }
    }
    return nullptr;
}

static void activation_quant_reuse_release(activation_quant_reuse_state & state) {
    for (auto entry = state.entries.rbegin(); entry != state.entries.rend(); ++entry) {
        GGML_ASSERT(entry->data != nullptr);
        GGML_ASSERT(entry->pool != nullptr);
        entry->pool->free(entry->data, entry->actual_size);
    }
    state.entries.clear();
}

} // namespace

void ggml_cuda_activation_quant_reuse_begin(ggml_backend_cuda_context & ctx) {
    if (!activation_quant_reuse_enabled()) {
        return;
    }
    activation_quant_reuse_state * state = activation_quant_reuse_find(ctx);
    if (state == nullptr) {
        activation_quant_reuse_states.emplace_back();
        state = &activation_quant_reuse_states.back();
        state->ctx = &ctx;
    }
    GGML_ASSERT(state->depth != 0 || state->entries.empty());
    state->depth++;
}

void ggml_cuda_activation_quant_reuse_end(ggml_backend_cuda_context & ctx) {
    activation_quant_reuse_state * state = activation_quant_reuse_find(ctx);
    if (state == nullptr || state->depth == 0) {
        return;
    }
    state->depth--;
    if (state->depth == 0) {
        activation_quant_reuse_release(*state);
    }
}

void * ggml_cuda_activation_quant_reuse_acquire(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src,
        ggml_cuda_activation_quant_type type,
        size_t size,
        bool & hit) {
    hit = false;
    activation_quant_reuse_state * state = activation_quant_reuse_find(ctx);
    if (state == nullptr || state->depth == 0 || ctx.curr_stream_no != 0) {
        return nullptr;
    }

    for (activation_quant_reuse_entry & entry : state->entries) {
        if (entry.src == src && entry.src_data == src->data &&
                entry.type == type && entry.size == size) {
            hit = true;
            return entry.data;
        }
    }

    state->entries.emplace_back();
    activation_quant_reuse_entry & entry = state->entries.back();
    entry.pool = &ctx.pool();
    entry.data = entry.pool->alloc(size, &entry.actual_size);
    entry.src = src;
    entry.src_data = src->data;
    entry.type = type;
    entry.size = size;
    return entry.data;
}
