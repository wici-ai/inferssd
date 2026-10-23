# Cloud 5090 Muse reference launch

`cloud-5090-muse-f5-tier-off-reserve1024.json` is a separate, unexecuted
static mapping for untouched f5 versus
`a1bd9d22248dc897e288af303cde9c501952b347`. Both sides disable the expert GPU
tier while retaining the model, 64 GiB host limit, CPU settings, dense GPU
offload and the explicit sixteen-setting snapshot. It records the five legacy
environment changes, the candidate's equivalent `vram.budget=0` derivation,
and exact source-file hashes and lines. It does not replace the historical
8 GiB draft below and cannot establish no regression for that mode.

The tier-off draft retains a 1024 MiB allocator reserve for the ready-context
request measurement. Candidate CUDA service attachment registers that reserve
independently of tier creation. There is a startup distinction: f5's environment
guard also covers model loading, whereas the candidate context registration
occurs afterward. The draft explicitly records this scope and an **unapplied**
reserve-zero alternative if equal startup coverage is required. It is not an
executable binding or job authorization; actual identities and acceptance of
the stated scope are still required before a device comparison.

`cloud-5090-muse-f5.json` is a source-reviewed launch draft for untouched
`f5e6e26ff26d2014cccef0118048039b88163f54`, compared with the actual behavior of
`677763d63974b073cfeb68de1e276d8ce513e493`. It contains a concrete argv/environment,
all fourteen settings, and SHA-256 of every cited source file. It is **not an
executable comparison binding or a successful benchmark receipt**. The final
candidate is changing; review its sixteen-key registry before binding it.

The JSON separately records the two-key extension reviewed against registry
`9c5596e9b1424d300b780996e509e0a86097a04c`: `model.engram_prefetch=0` and
`model.hc_fused=1`. f5's `src/models/deepseek41.cpp` gives exactly these defaults
when `LLAMA_DSV41_ENGRAM_PREFETCH` and `LLAMA_DSV41_HC_FUSED` are absent. Neither
changes Muse execution. Construct a complete explicit sixteen-key final binding;
do not add these fields to historical fourteen-key telemetry. The adapter accepts
either complete generation, prefers the installed `inferssd` registry, and rejects
a partial generation or a sixteen-key binding with only a fourteen-key registry.
Build/template bindings may cover just `muse` for this checkpoint.

The draft uses the agreed 22-shard Muse Q4_K_S source, alias paths under each
sealed build, model template under each source tree, port 18080, 10 total CPU
participants, context 1024, batch/ubatch 512, no warmup, and no H2D cap. Both sides
must use identical common server options, the pinned short prompt, 256 generated
tokens, host memory limit 64 GiB, no swap, and the same entire allowed CPU set.
The explicit single-slot/cache options must also be copied to the candidate.

Build f5 targets `llama-server` **and `kimi-vram-tier`**. Its Muse tier is a
separate DSO, loaded by the recorded `LD_PRELOAD`; Q4_K/Q5_K use its grouped
adapter, not the MXFP4 parity library. Do not add the old mmap preload shim,
`LLAMA_DROP_*`, prefill copy switches, or an H2D governor merely because the old
launcher set them. The draft follows the candidate's ambient-default mmap
behavior and explicit CPU-sink path. A changed candidate launch needs another
source review of those specific options.

Two resource details prevent a blind old-name/new-name translation:

- In 677, `vram.reserve_mib=1024` reaches the tier's low watermark. The CUDA
  allocator still reads only `GGML_CUDA_MEMORY_RESERVE_MIB`; the typed setting
  does not supply its fixed reserve, and the dynamic setter is used only on
  unified memory. The draft therefore leaves that environment variable absent
  on both sides: actual fixed allocator reserve **0**, low watermark **1024**,
  startup/elastic/high margin **1088** MiB. If the final candidate fixes the
  typed allocator bridge, set the old variable to `1024` and regenerate the
  source/profile binding. Do not claim the current draft enforces a 1024 MiB
  allocator reserve.
- The new process arbiter applies a 2048 MiB system margin on this discrete GPU.
  f5's `VRAM_HOST_HEADROOM_MIB` is ignored there. The draft retains f5's 512 MiB
  cgroup reserve and adds `GGML_KIMI_HOST_RESERVE_MIB=1536`, producing the same
  2048 MiB system floor. This adapter is specific to this discrete GPU, where
  neither side learns a unified-memory margin; it must not be reused for Spark.

Absent settings are intentional: old decode participants must be **unset**, not
`0` (which f5 rejects); both placement-cost spellings must be unset to retain the
identical priors. Empty CPU lists mean no explicit per-participant placement.
The Darwin cache default remains 4096 in the complete snapshot but has no Linux
effect. Resource-derived host cache and elastic device growth remain enabled;
the 8 GiB tier budget is not a hard VRAM cap.

Before acceptance, pin the final source SHA and exact raw profile, verify both
aliases have identical content and resolve to the same shard identities, and
fill actual loaded-image hashes and explicit roles. Copy the reviewed mapping
into `reference.py`'s binding only after resolving the resource caveat. Never
invent those hashes or typed telemetry for f5. Compare the real raw SSE results
for prefill, TTFT, both decode halves and all 256 tokens. Any observed slowdown
blocks acceptance as inconclusive; this checkpoint is one Muse/512 cell, not
the full device/model/prompt matrix.
