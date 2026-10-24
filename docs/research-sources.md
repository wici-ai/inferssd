<!-- SPDX-License-Identifier: MIT -->
# Research source registry

Updated: 2026-10-24. Target: SSDash / InferSSD.
This records manual research coverage; it does not configure a scheduled scanner.
Review means selected source paths and patches, not a whole-repository audit.

| Source | Focus | Current status |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Model compatibility, vocabulary loading, graphs and kernels | Selective vocabulary and Nemotron dimension ports prepared; see the October 23 report |
| [Strata](https://github.com/Niko1221/Strata) | Expert indexing, Metal ownership, prefill overlap, measured scheduling costs | Indexing candidate integrated into this branch; broader designs pending |
| [Colibri](https://github.com/JustVugg/colibri) | Persistent expert history, admission hysteresis, scheduling | Selected implementations reviewed; no implementation copied |
| [MoE-Infinity](https://github.com/TorchMoE/MoE-Infinity) | Prefetch budgets and computation/transfer overlap | Selected implementations reviewed; deferred pending workload evidence |
| [HybriMoE](https://github.com/PKU-SEC-Lab/HybriMoE) | Routing-score smoothing and prefetch/cache interaction | Selected implementations reviewed; overlap with existing policies requires comparison |
| [KTransformers](https://github.com/kvcache-ai/ktransformers) | CPU/GPU expert placement and heterogeneous computation | Selected implementations reviewed; no port selected |
| [FreeToken](https://github.com/FlashML-org/FreeToken) | Bandwidth-adaptive CPU/GPU execution, expert caching, prefill streaming and memory budgets | Added October 24; initial README and cache/prefill source inspection complete; no port or benchmark yet |

## FreeToken: initial inspection

Pinned revision:
[`32056cf4608b07c21afc45fa510c74fba67dcf17`](https://github.com/FlashML-org/FreeToken/commit/32056cf4608b07c21afc45fa510c74fba67dcf17).
Repository license: Apache-2.0. No FreeToken code is incorporated in this change.
Any later code port must retain applicable notices and component licensing.

Inspected sources:

- [README](https://github.com/FlashML-org/FreeToken/blob/32056cf4608b07c21afc45fa510c74fba67dcf17/README.md)
- [Expert offload cache](https://github.com/FlashML-org/FreeToken/blob/32056cf4608b07c21afc45fa510c74fba67dcf17/python/freetoken/moe/offload_cache.py)
- [Prefill scheduler](https://github.com/FlashML-org/FreeToken/blob/32056cf4608b07c21afc45fa510c74fba67dcf17/python/freetoken/scheduler/prefill.py)

Priority questions for a deeper transfer review:

1. **Reuse hits during prefill.** `prefill_hit_d2d` gathers already resident
   experts on the device and transfers misses, with explicit buffer/layout
   requirements and a full-layer fallback. Compare useful bytes and synchronization
   cost against SSDash's current pack/alias prefill path before selecting a port.
2. **Balance miss computation and transfers.** The hybrid cache path can fetch a
   subset of misses while CPU workers compute the remainder. Its per-layer fetch
   fraction is described in terms of measured transfer/CPU bandwidth. Inspect the
   calibration and actual scheduling code next; compare with existing SSDash
   cost policies rather than assuming this is a missing feature.
3. **Overlap safely.** Prefill uses two staging buffers, a copy stream, and
   release/ready ordering. `begin_prefill` fences earlier compute before staging
   can overwrite expert slots. Evaluate this ownership protocol together with
   SSDash's pinning and prefill isolation.
4. **Account for multiple memory pools.** The cache exposes validation/rebuild
   paths; prefill admission accounts for additional KV pages and hybrid state
   slots. Follow the engine's memory coordinator before assessing runtime expert/KV
   rebudgeting. The inspected methods alone do not establish end-to-end behavior.
5. **Follow model additions.** Compare model registries, weight layouts and
   architecture-specific kernels in a subsequent pass. llama.cpp remains the
   primary source for GGUF/runtime model ports; a FreeToken model announcement
   does not establish compatible GGUF support.

The inspected execution path uses PyTorch CUDA streams and device/host transfer
mechanisms. Transfer to Linux CUDA should be evaluated first. Apple unified
memory needs separate cost and synchronization analysis; this review neither
establishes native Metal support nor proves that CUDA copy strategies will help
on a Mac. It also does not establish benefits for SSD-resident experts from
host-memory-to-GPU measurements alone.

Prior evidence: [October 23 performance and model review](upstream-research-2026-10-23.md).
The [first-pass cache-index review](cache-index-design-review.md) and its code are
now integrated into this branch. The original `mac-cache-index` branch remains
available. See the [main integration report](research-integration-2026-10-24.md)
for current paths and validation. Adding a research source does not establish a
throughput gain or enable an additional cache policy.
