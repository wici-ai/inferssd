# Cloud 5090 Muse reference launch drafts

These files are static, source-reviewed launch mappings between the legacy
reference revision (`LEGACY_REVISION` in `../reference.py`, abbreviated `f5` in
the file names) and a typed-settings candidate for one Muse 512/256 cell on
`cloud-5090`. Neither file has been executed. Neither is an executable
comparison binding, a job authorization or a benchmark receipt. Each JSON
records its candidate revision in `reviewed_candidate_revision` and its open
items in `unresolved`.

- `cloud-5090-muse-f5.json` maps an 8 GiB expert GPU tier configuration. It
  contains a concrete argv/environment, all fourteen settings of its registry
  generation, and the SHA-256 of every cited source file.
- `cloud-5090-muse-f5-tier-off-reserve1024.json` disables the expert GPU tier on
  both sides while retaining the model, 64 GiB host limit, CPU settings, dense
  GPU offload and an explicit sixteen-setting snapshot. It records the five
  legacy environment changes, the candidate's equivalent `vram.budget=0`
  derivation, and exact source-file hashes and lines. It does not establish
  anything about the tiered configuration.

## Shared launch assumptions

Both drafts use the 22-shard Muse Q4_K_S source, alias paths under each
read-only build, the model template under each source tree, port 18080, 10
total CPU participants, context 1024, batch/ubatch 512, no warmup and no H2D
cap. Both sides must use identical common server options, the pinned short
prompt, 256 generated tokens, a 64 GiB host memory limit, no swap and the same
allowed CPU set. The explicit single-slot/cache options must also be copied to
the candidate.

The legacy build targets `llama-server` **and `kimi-vram-tier`**. Its Muse tier
is a separate DSO loaded through the recorded `LD_PRELOAD`; Q4_K/Q5_K use its
grouped adapter, not the MXFP4 parity library. Do not add the old mmap preload
shim, `LLAMA_DROP_*`, prefill copy switches or an H2D governor merely because
the old launcher set them. The drafts follow the candidate's ambient-default
mmap behavior and explicit CPU-sink path. A changed candidate launch needs
another source review of those options.

The sixteen-key registry adds `model.engram_prefetch=0` and `model.hc_fused=1`.
The legacy `src/models/deepseek41.cpp` gives exactly these defaults when
`LLAMA_DSV41_ENGRAM_PREFETCH` and `LLAMA_DSV41_HC_FUSED` are absent, and neither
changes Muse execution. Construct a complete explicit sixteen-key binding; do
not add these fields to fourteen-key telemetry. The adapter accepts either
complete generation, prefers the installed `inferssd` registry, and rejects a
partial generation or a sixteen-key binding with only a fourteen-key registry.
Build/template bindings may cover just `muse` for this checkpoint.

## Resource mapping

Two resource details prevent a blind old-name/new-name translation:

- In the 8 GiB draft's candidate, `vram.reserve_mib=1024` reaches the tier's low
  watermark, but the CUDA allocator still reads only
  `GGML_CUDA_MEMORY_RESERVE_MIB`; the typed setting does not supply its fixed
  reserve, and the dynamic setter is used only on unified memory. The draft
  therefore leaves that variable absent on both sides: actual fixed allocator
  reserve **0**, low watermark **1024**, startup/elastic/high margin **1088**
  MiB. If the candidate connects the typed allocator reserve, set the old
  variable to `1024` and regenerate the binding. The draft does not enforce a
  1024 MiB allocator reserve.
- The candidate's process arbiter applies a 2048 MiB system margin on this
  discrete GPU, where the legacy `VRAM_HOST_HEADROOM_MIB` is ignored. The drafts
  retain the legacy 512 MiB cgroup reserve and add
  `GGML_KIMI_HOST_RESERVE_MIB=1536`, producing the same 2048 MiB system floor.
  This adapter is specific to a discrete GPU, where neither side learns a
  unified-memory margin; do not reuse it for DGX Spark.

The tier-off draft retains a 1024 MiB allocator reserve for the ready-context
request measurement. Candidate CUDA service attachment registers that reserve
independently of tier creation. The legacy environment guard also covers model
loading, whereas the candidate's context registration occurs afterward. The
draft records this scope and an **unapplied** reserve-zero alternative if equal
startup coverage is required.

Absent settings are intentional: old decode participants must be **unset**, not
`0` (which the legacy launcher rejects); both placement-cost spellings must be
unset to retain identical priors. Empty CPU lists mean no explicit
per-participant placement. The Darwin cache default remains 4096 in the
complete snapshot but has no Linux effect. Resource-derived host cache and
elastic device growth remain enabled; the 8 GiB tier budget is not a hard VRAM
cap.

## Before use

Pin the executed candidate SHA and exact raw profile, verify both aliases have
identical content and resolve to the same shard identities, and fill actual
loaded-image hashes and explicit roles. Copy the reviewed mapping into
`reference.py`'s binding only after resolving the resource caveats above. Never
invent those hashes or typed telemetry for the legacy side. Compare the real raw
SSE results for prefill, TTFT, both decode halves and all 256 tokens. Any
observed slowdown blocks acceptance as inconclusive; one Muse/512 cell is not
the full device/model/prompt matrix.
