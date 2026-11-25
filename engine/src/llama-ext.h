#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP

#include "llama.h"

#include <cstdint>
#include <map>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// Draft-model staging APIs. These remain in llama-ext until the speculative API stabilizes.
LLAMA_API void llama_set_embeddings_nextn(
        struct llama_context * ctx, bool value, bool masked);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

LLAMA_API void llama_set_embeddings_layer_inp(
        struct llama_context * ctx, uint32_t lid, bool value);
LLAMA_API float * llama_get_embeddings_layer_inp(
        struct llama_context * ctx, uint32_t lid);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

LLAMA_API const int32_t * llama_model_target_layer_ids(const struct llama_model * model);
LLAMA_API uint32_t llama_model_target_layer_ids_n(const struct llama_model * model);

LLAMA_API uint32_t llama_model_dspark_block_size(const struct llama_model * model);
LLAMA_API bool llama_model_inkling2_dspark_attach_target(struct llama_model * model_dft, const struct llama_model * model_tgt);
LLAMA_API void llama_model_inkling2_dspark_set_f_rnn(struct llama_model * model_dft, const float * f_rnn);
LLAMA_API void llama_model_inkling2_dspark_set_pending_taps(struct llama_model * model_dft, const float * taps, int32_t n);
LLAMA_API void llama_model_inkling2_dspark_set_cum_min(struct llama_model * model_dft, float cum_min);
LLAMA_API void llama_model_inkling2_dspark_set_tree_params(
        struct llama_model * model_dft, float trunk_min, float alt_min);
LLAMA_API void llama_model_inkling2_dspark_get_draft_ids(const struct llama_model * model_dft, int32_t * dst, uint32_t n);
LLAMA_API void llama_model_inkling2_dspark_get_draft_ids_and_probs(
        const struct llama_model * model_dft, int32_t * dst_ids, float * dst_probs, uint32_t n);
LLAMA_API void llama_model_inkling2_dspark_get_tree(
        const struct llama_model * model_dft, int32_t * dst_ids, float * dst_probs, uint32_t n);

// Restrict drafting to n_slots tokens: the n_ascii ASCII and n_other other tokens with the largest folded Markov bias,
// the control tokens, and room for tokens added later. Verification is unchanged, so a token outside the set only
// cannot be drafted. Call after llama_model_inkling2_dspark_attach_target.
LLAMA_API bool llama_model_inkling2_dspark_init_draft_vocab(
        struct llama_model * model_dft, uint32_t n_ascii, uint32_t n_other, uint32_t n_slots);
// Add tokens to the draft vocabulary while slots are free; returns the number added. Call while no draft is computing.
LLAMA_API uint32_t llama_model_inkling2_dspark_add_draft_vocab(
        struct llama_model * model_dft, const int32_t * ids, uint32_t n);
