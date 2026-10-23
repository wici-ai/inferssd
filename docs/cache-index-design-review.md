<!-- SPDX-License-Identifier: MIT -->
# Mac expert cache indexing: design review

This candidate is based on SSDash `f5e6e26` (2026-10-23). It reduces metadata
search work in the existing Mac expert reader; it does not introduce a new
eviction policy, change expert routing, or move expert computation to Metal.
It is a local candidate, not an accepted performance optimization.

## First research pass

The review examined the implementations at the following fixed revisions.
Upstream measurements are not measurements of this SSDash candidate.

| Source | Revision | Finding and decision |
| --- | --- | --- |
| [Strata](https://github.com/Niko1221/Strata) | `011c9e600a93bbc9fd451c563501b5831c32bbca` | Explicit expert-to-slot indexing is useful for the Mac cache's metadata path. Selected. Its shared Metal expert arena is a separate, much larger project. |
| [Colibri](https://github.com/JustVugg/colibri) | `e0db083cdc9e563e0766cc6fc661ef23c32298e0` | Persistent routing history and admission hysteresis merit later workload-specific study. Cache-aware routing changes expert selection and is excluded from this numerical-preserving change. |
| [MoE-Infinity](https://github.com/TorchMoE/MoE-Infinity) | `7ff99b101b995ec087fde7fccb20b56bdc9b5a64` | A computation-window/transfer-budget controller could extend SSDash's fixed-width prefetch. Deferred: its own source also records a route-ahead priority experiment without a robust throughput win. |
| [HybriMoE](https://github.com/PKU-SEC-Lab/HybriMoE) | `508a85b9a43dff37124370d8816e411e74827803` | Routing-score smoothing and prefetch overlap partly overlap existing policies. A replacement needs evidence against the current cost/decay policies. Deferred. |
| [KTransformers](https://github.com/kvcache-ai/ktransformers) | `a694e718e8b3ec2e221277d674e369f3be38d2a4` | CPU/GPU expert placement overlaps the existing Linux implementation; copying the framework is not justified. Deferred. |

Selected source locations:

- [Strata slot map and ownership](https://github.com/Niko1221/Strata/blob/011c9e600a93bbc9fd451c563501b5831c32bbca/metal/src/engine/stream.h)
- [Strata slot and LRU operations](https://github.com/Niko1221/Strata/blob/011c9e600a93bbc9fd451c563501b5831c32bbca/metal/src/engine/stream.cpp#L226)
- [MoE-Infinity overlap budget](https://github.com/TorchMoE/MoE-Infinity/blob/7ff99b101b995ec087fde7fccb20b56bdc9b5a64/moe_infinity/memory/overlap_budget.py)
- [MoE-Infinity negative experiment](https://github.com/TorchMoE/MoE-Infinity/blob/7ff99b101b995ec087fde7fccb20b56bdc9b5a64/core/prefetch/archer_prefetch_handle.cpp#L319)

SSDash already implements decaying caches, optional next-layer prefetch and
in-flight handling in `kimi-cpu-sink.cpp`; its CUDA tier also has cost-based
admission and elastic memory management. These are not new features to import.
The Mac reader has its own cache and selected-expert read-ahead path.

## Implemented scope

Previously, each miss scanned the entire slot table to count the admitted
slices of its tensor and select a local victim, even when admission was
disabled. Each release scanned the table to find the owning payload pointer.

The candidate adds two non-owning indexes:

1. Payload address to slot entry: average constant-time lookup on release.
2. Tensor name to admitted slot entries: constant-time quota count and a victim
   scan limited to that tensor, performed only when its quota is full.

The global budget fallback still uses the original full-table scan. Removing
an admitted entry from the per-tensor vector takes linear time in that tensor's
admitted entries. This change does not claim every cache operation is O(1).

The implementation retains LFU heat/decay, LRU recency, admission ties, pinning,
prefill isolation, scratch reuse, read-ahead order and GGUF/pack payload bytes.
Admitted entries have distinct decode recency stamps, so changing the local
iteration order does not change the victim tie-break. No upstream code was
copied; the existing SSDash implementation was adapted to the indexing idea.

## Ownership and resource accounting

`slots` remains the sole owner of each payload. Indexes store pointers to its
entries, not unordered-map iterators: element pointers survive rehash. All index
operations use the existing cache mutex. Every removal clears the aliases before
freeing or recycling storage. A failed index allocation removes the unpublished
slot and any newly created empty index group before propagating the exception.

The payload index includes transient slices while they are pinned. The tensor
index includes only admitted slices. The final transient release removes its
payload index before returning memory to the scratch buffer. No worker threads
or external file formats are added.

Metadata grows with the number of live slices and admitted tensors: an additional
hash entry per live payload and a pointer per admitted slice, plus hash buckets
and vector capacity. Weight buffers are not duplicated. As in the baseline,
the cache byte statistic measures payload allocations, not total process RSS;
metadata and simultaneously pinned transient slices require separate headroom.

## Validation and acceptance

The expanded portable test uses 96 tensors with four experts each, forces map
growth with live pins, releases in a different order, checks nested/double/unknown
releases, and exercises repeated eviction and scratch reuse under both LFU and
LRU. A second fixture changes backing-file bytes to distinguish cache retention
from rereading, verifying per-tensor eviction decisions rather than only output
addresses. Existing concurrent prefill/decode tests remain in place.

Build and run the relevant tests using the normal CMake targets:

```sh
cmake --build build-mac --target llama-server test-mac-expert-pack-cache test-mac-expert-source test-mac-expert-operator
ctest --test-dir build-mac --output-on-failure -R '^(mac_expert_pack_cache|mac_expert_operator_.*)$'
python3 tests/test-mac-expert-source.py build-mac/bin/test-mac-expert-source
```

For a device comparison, keep the model, routing, prompts, resource budgets,
cache policy, read depth and input/output token counts fixed. Compare
`policy_us`, `release_us` and lock time alongside prefill/decode throughput and
process memory. Cache hits, reads and evictions should remain unchanged for a
deterministic serial access trace. Timing and concurrent scheduling can differ.

This change is useful only to the extent metadata management is exposed on the
critical path. Lower synthetic metadata time is not a claim of model speedup.
Real-model performance and Linux execution remain separate device checks.

## Local validation record (2026-10-23)

- Apple Silicon Release build with Metal and Accelerate: `llama-server`, cache,
  source and expert-operator targets built successfully.
- Portable cache regressions: passed, including the added multi-tensor cases.
- AddressSanitizer + UndefinedBehaviorSanitizer: passed for cache regressions.
- ThreadSanitizer: passed for cache regressions, including concurrent batches.
- Expert source fixtures: all 78 cases passed (pack, GGUF aliases, alignment,
  bounded read-ahead, short reads, pinning and prefill isolation).
- Framework Python tests: 108 passed. Mac launcher Python tests: 16 passed.
- Production CPU-hook operator checks: 39 CTest cases passed before stopping
  the long-running 12-thread case. The varying-thread-count case subsequently
  passed in a direct run against both base and candidate (14.42 s and 14.35 s).
  The fixed 12-thread case exceeded a separate 15 s diagnostic deadline on both
  base and candidate. The candidate also exceeded a minute in the earlier run.
  This case is unresolved, not passing; the diagnostic deadline is shorter than
  the repository's normal 180 s test timeout. These checks ran on an 8-core
  Apple M2 with 8 GiB unified memory, macOS 26.6.2.
- The same four new deterministic fixtures also passed against the unmodified
  base reader. Requests, hits, misses, read requests/bytes, evictions, scratch
  reuse, cache bytes and peak cache bytes matched the candidate exactly.
- Linux shell launcher suites were attempted but failed on this Mac: the system
  Bash lacks associative arrays, and cgroup/Linux prerequisites are absent.
  These are pending Linux checks, not passes or accepted skips.

The live-model 512/4K comparisons have not been run. Publication and PR creation
are intentionally outside this local implementation step.
