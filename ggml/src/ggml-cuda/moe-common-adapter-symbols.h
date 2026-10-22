#pragma once

// Keep the restored common adapter link-isolated from the Kimi-K3-local
// implementation.  The common translation units include this file before
// their historical implementation so both variants can coexist in one CUDA
// backend.  Public dispatch remains on the unprefixed symbols.

#define ggml_cuda_host_register                                      ggml_cuda_common_host_register
#define ggml_cuda_moe_stream_available                               ggml_cuda_moe_common_stream_available
#define ggml_cuda_moe_stream_ds4_triplet_host_role_ptr               ggml_cuda_moe_common_stream_ds4_triplet_host_role_ptr
#define ggml_cuda_moe_stream_link_anchor                             ggml_cuda_moe_common_stream_link_anchor
#define ggml_cuda_moe_stream_one                                     ggml_cuda_moe_common_stream_one
#define ggml_cuda_moe_stream_one_cache_contains                      ggml_cuda_moe_common_stream_one_cache_contains
#define ggml_cuda_moe_stream_one_cache_dev_ptr_or_insert             ggml_cuda_moe_common_stream_one_cache_dev_ptr_or_insert
#define ggml_cuda_moe_stream_one_cache_dev_ptrs_or_insert            ggml_cuda_moe_common_stream_one_cache_dev_ptrs_or_insert
#define ggml_cuda_moe_stream_one_cache_ready                         ggml_cuda_moe_common_stream_one_cache_ready
#define ggml_cuda_moe_stream_one_cache_release_descriptors           ggml_cuda_moe_common_stream_one_cache_release_descriptors
#define ggml_cuda_moe_stream_one_cache_release_descriptors_synchronized \
    ggml_cuda_moe_common_stream_one_cache_release_descriptors_synchronized
#define ggml_cuda_moe_stream_one_gate_up_prefetch                    ggml_cuda_moe_common_stream_one_gate_up_prefetch
#define ggml_cuda_moe_stream_q80_hot_batch_probe                     ggml_cuda_moe_common_stream_q80_hot_batch_probe
#define ggml_cuda_moe_stream_q80_probe                               ggml_cuda_moe_common_stream_q80_probe
#define ggml_cuda_moe_stream_q80_skip                                ggml_cuda_moe_common_stream_q80_skip
#define ggml_cuda_moe_stream_q80_write                               ggml_cuda_moe_common_stream_q80_write
#define ggml_cuda_moe_stream_sync                                    ggml_cuda_moe_common_stream_sync

#define ggml_cuda_moe_expert_pack_mmap_ptr                           ggml_cuda_moe_common_expert_pack_mmap_ptr
#define ggml_cuda_moe_expert_pack_mmap_ptr_debug                     ggml_cuda_moe_common_expert_pack_mmap_ptr_debug
#define ggml_cuda_moe_expert_pack_mmap_ptr_source_debug              ggml_cuda_moe_common_expert_pack_mmap_ptr_source_debug
#define ggml_cuda_moe_expert_pack_read_to_host                       ggml_cuda_moe_common_expert_pack_read_to_host
#define ggml_cuda_moe_expert_pack_resolve_direct                     ggml_cuda_moe_common_expert_pack_resolve_direct
#define ggml_cuda_moe_expert_pack_copy_ranges_to_device              ggml_cuda_moe_common_expert_pack_copy_ranges_to_device
#define ggml_cuda_moe_expert_pack_v2_lookup_debug                    ggml_cuda_moe_common_expert_pack_v2_lookup_debug
#define ggml_cuda_moe_expert_pack_v2_read_debug                      ggml_cuda_moe_common_expert_pack_v2_read_debug
#define ggml_cuda_moe_iq2_prompt_replay                              ggml_cuda_moe_common_iq2_prompt_replay
#define ggml_cuda_moe_iq2_q8k_r8_selftest                            ggml_cuda_moe_common_iq2_q8k_r8_selftest
#define ggml_cuda_moe_iq2_xxs_q8k_selftest                           ggml_cuda_moe_common_iq2_xxs_q8k_selftest
#define ggml_cuda_moe_stream_batch                                   ggml_cuda_moe_common_stream_batch
#define ggml_cuda_moe_stream_decode_batch                            ggml_cuda_moe_common_stream_decode_batch
#define ggml_cuda_moe_stream_batch_flush_gate_updown_cosubmit        ggml_cuda_moe_common_stream_batch_flush_gate_updown_cosubmit
#define ggml_cuda_moe_stream_batch_link_anchor                       ggml_cuda_moe_common_stream_batch_link_anchor
#define ggml_cuda_moe_stream_batch_phase_report                      ggml_cuda_moe_common_stream_batch_phase_report
#define ggml_cuda_moe_stream_batch_preload_active_from_pack          ggml_cuda_moe_common_stream_batch_preload_active_from_pack
#define ggml_cuda_moe_stream_batch_preload_gate_updown               ggml_cuda_moe_common_stream_batch_preload_gate_updown
#define ggml_cuda_moe_stream_batch_report_counters                   ggml_cuda_moe_common_stream_batch_report_counters
#define ggml_cuda_moe_stream_cache_contains                          ggml_cuda_moe_common_stream_cache_contains
#define ggml_cuda_moe_stream_cache_dev_ptr                           ggml_cuda_moe_common_stream_cache_dev_ptr
#define ggml_cuda_moe_stream_handoff_upload                          ggml_cuda_moe_common_stream_handoff_upload
#define ggml_cuda_moe_stream_preload_expert_async                    ggml_cuda_moe_common_stream_preload_expert_async
#define ggml_cuda_moe_stream_preload_expert_from_pack_async          ggml_cuda_moe_common_stream_preload_expert_from_pack_async
#define ggml_cuda_moe_stream_preload_synchronize                     ggml_cuda_moe_common_stream_preload_synchronize
#define ggml_cuda_moe_stream_preload_tensor                          ggml_cuda_moe_common_stream_preload_tensor
#define ggml_cuda_moe_stream_preload_tensor_async                    ggml_cuda_moe_common_stream_preload_tensor_async
#define ggml_cuda_moe_stream_preload_tensor_prompt                   ggml_cuda_moe_common_stream_preload_tensor_prompt
#define ggml_cuda_moe_stream_q4_colibri_pilot_prefetch_from_gate     ggml_cuda_moe_common_stream_q4_colibri_pilot_prefetch_from_gate
#define ggml_cuda_moe_stream_register_tensor                         ggml_cuda_moe_common_stream_register_tensor
#define ggml_cuda_moe_stream_route_group_preload_updown_from_gate    ggml_cuda_moe_common_stream_route_group_preload_updown_from_gate
#define ggml_cuda_moe_stream_stage_active_experts                    ggml_cuda_moe_common_stream_stage_active_experts
#define ggml_cuda_moe_stream_up_gate_batch                           ggml_cuda_moe_common_stream_up_gate_batch
#define ggml_cuda_moe_ttft_trace_event                               ggml_cuda_moe_common_ttft_trace_event
#define ggml_cuda_moe_ttft_trace_mark                                ggml_cuda_moe_common_ttft_trace_mark
