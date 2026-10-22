# Experimental Kimi CPU sink VRAM tier

Linux, one sequence, decode only. Disabled unless a positive
`GGML_KIMI_CPU_SINK_VRAM_BYTES` is set. This private experiment keeps the
original expert list, Q3_K weights, Q8_K activations and slot-order sum.

Build the optional DSOs with `-DGGML_CUDA_KIMI_VRAM_TIER=ON`, then build the
`kimi-q3k-parity` and `kimi-vram-tier` targets. The server must also contain
the CPU sink adapter and single-token reserve change. Explicitly preload
both DSOs, preserving any existing H2D governor and mmap shim.
The parity target compiles test2's frozen v7 chain translation unit, outside
the backend CUDA flags, with `--fmad=false --ftz=false`. The chain includes
the base parity implementation; neither unit enters the backend fast-math
source list. The experimental deployment uses test2's `6d97f987...` binary.
After a component build or hardware change, rerun its
1,521 operator checks before the full-model comparison.

```
GGML_KIMI_CPU_SINK_DECODE_RESERVE=1
GGML_KIMI_CPU_SINK_READY_FIRST=1
GGML_KIMI_Q3K_GPU_PARITY=1
GGML_KIMI_CPU_SINK_VRAM_BYTES=4294967296
GGML_KIMI_CPU_SINK_VRAM_POLICY=count_decay2
GGML_KIMI_CPU_SINK_VRAM_MIN_COUNT=2.5
GGML_KIMI_CPU_SINK_VRAM_SAFETY_MIB=512
GGML_KIMI_CPU_SINK_VRAM_ELASTIC=0
GGML_KIMI_CPU_SINK_VRAM_ELASTIC_SAFETY_BLOCKS=0
GGML_KIMI_CPU_SINK_VRAM_ELASTIC_SAFETY_MIB=32
GGML_KIMI_CPU_SINK_VRAM_EXCLUSIVE=0
GGML_KIMI_CPU_SINK_VRAM_ASYNC_UPLOAD=0
GGML_KIMI_CPU_SINK_VRAM_GPU_HIDDEN=0
```

`VRAM_SAFETY_MIB` participates in the original arena's initial sizing.
`VRAM_ELASTIC_SAFETY_BLOCKS` and `VRAM_ELASTIC_SAFETY_MIB` are separate
decode-growth holdbacks.

`VRAM_RESERVE_MIB` of the DeepSeek V4.1 launcher (default 0) leaves GPU headroom to
another process that stays on the card, such as a deployment's RPC listener. It
sets `GGML_CUDA_MEMORY_RESERVE_MIB` to that amount and raises the decode watermarks
from 16/64 MiB to the reserve and 64 MiB above it; 576 leaves at least 500 MiB plus
a margin for late CUDA graph/kernel allocations, with 576/640 MiB watermarks.
`GGML_CUDA_MEMORY_RESERVE_MIB` checks physical free memory before device-buffer
and VMM pool allocations. It reclaims quiescent elastic expert blocks before a
prefill workspace can consume the reserve. If reclaim is unavailable, allocation
fails normally so the existing graph planner can reduce its batch. The backend
limit defaults to zero for other launchers; the expert tier's own growth and
scratch allocations continue to use its safety margins and watermarks.

These are experimental settings, not a validated speed recommendation.
The default budget is zero; the default minimum count is zero. LRU requires
minimum count zero. The count is a per-layer selection frequency with a
two-token half-life; duplicate IDs count once. Uniform quotas use the same
92-layer priority list as the RAM cache. Allocation is clamped to available
CUDA memory after the safety margin. There is no silent CPU fallback when
an enabled GPU component fails.

A bridge error closes the hidden chain and base handle before returning to
the cache layer. Component destruction drains queued computation even when
submission or wait failed. The failed bridge stays closed; it cannot retry
or publish a later result. If the stream cannot drain, the process aborts
while its cache leases are still held.

All hits are pinned before admissions are planned. A triplet becomes valid
only after its upload completes. RAM slots can be borrowed through the
entire layer; ring sources must finish copying before their slot is reused.
Both the normal finish and the error/cancel path drain uploads before RAM
cache release. Prefill and request-epoch changes release the weight arena.

The dedicated worker runs the test2 component's two GPU matvec stages with
a host SiTU/Q8_K bridge between them. It allocates at most 32 small matvec
jobs, never an operator staging tensor for all 896 experts. GPU results are
unweighted independent rows; the CPU adapter performs the separately
rounded weight multiplication and original ordered sum.

Five independent options default to zero:

- `VRAM_EXCLUSIVE=1` releases a duplicate RAM entry after all CPU work, read
  CQEs and uploads drain. A separate residency ABI pins the completed GPU copy
  using the model epoch, complete packed-file identities and a monotonic
  confirmation generation. Any live RAM lease vetoes the drop. Touched pages
  remain committed and can be reused without allocating another payload.
- `VRAM_ASYNC_UPLOAD=1` queues borrowed RAM pointers to the same worker before
  CPU expert computation. A bounded queue holds at most 16 uploads per layer.
  The layer finish waits for the remainder; ring pointers still cannot be
  reused until copied. This does not pipeline uploads across layer boundaries.
- `VRAM_DEFERRED_UPLOAD=1` (requires `VRAM_ASYNC_UPLOAD=1`) pipelines borrowed
  RAM uploads across layer boundaries. A dedicated thread copies them on its
  own CUDA stream and the layer finish no longer waits. The reserved GPU slot is
  marked uploading and is never a victim until `kimi_vram_deferred_api` collect
  publishes it after the copy reached the device; the CPU owner keeps the RAM
  source leased until then and, with `VRAM_EXCLUSIVE=1`, drops the RAM duplicate
  afterwards. The owner collects a layer's uploads before that layer's next RAM
  prepare, every upload at a prefill, epoch or token discontinuity, and waits
  for all of them before the RAM cache unmaps its arena (pressure pause or
  geometry change). Release, cancel and destroy wait for in-flight copies. At
  most 32 copies are in flight; a further admission waits for one to finish.
  Ring sources keep the synchronous path. Not available with the cache-location
  provider.
- `VRAM_ELASTIC=1` splits a retained global pool into blocks, shrinks decode-only
  blocks before the next prefill reserve, and grows again after the decode
  reserve. A backend CUDA allocation that runs out of memory invokes the armed
  reclaimer, releases one tail block and retries; once decode-grown blocks are
  gone it may continue below the policy base. `VRAM_ELASTIC_SAFETY_BLOCKS` and
  `VRAM_ELASTIC_SAFETY_MIB` hold back decode growth for allocations that bypass
  that backend hook. The byte default is 32 MiB, selected for the validated
  DeepSeek-V4.1 profile from about 21 MiB of observed late CUDA graph
  allocations; another device, graph or context must remeasure it. Elastic
  storage is refused for non-global pools, so the Kimi K3 path keeps the
  original contiguous arena.
- `VRAM_GPU_HIDDEN=1` uses test2's GPU SiTU/Q8_K chain at the K3 3584/3072
  shape. Its quantizer explicitly multiplies then adds, following actual153.
  The base and chain must come from one DSO; older `8b2a7b9c...` components can
  only run the host bridge. Only final down vectors return to the host.

The first exclusive implementation retains the existing heat policy and
per-layer quotas. It has no D2H demotion or GPU persistence through prefill.
Its theoretical 926 RAM plus 302 GPU slots do not imply 1,228 occupied unique
experts at every step. Enable the new options as a speed recommendation only
after the same-boot comparison and full-output checks complete.

Two additional placement controls default to zero:

- `GGML_KIMI_CPU_SINK_VRAM_GLOBAL_POOL=1` lets all layers use the same arena
  slots. A victim's frequency is aged to the current token before comparing
  layers. Existing payload identities and compute/readiness leases apply.
- `GGML_KIMI_CPU_SINK_VRAM_DECAY_HALF_LIFE` accepts a finite value from 1 to
  64 tokens; zero or unset preserves the original two-token half-life. A
  nondefault half-life requires `count_decay2`. This changes cache admission,
  not the selected experts or their weights. The minimum count remains a
  separate setting, in the existing range 0 through 4.

`GGML_KIMI_CPU_SINK_VRAM_MEMORY_TRACE=1` samples CUDA free bytes at completed
expert-layer boundaries and around arena allocation/release. It defaults to
zero and must remain off for speed measurements. Its samples do not measure
transient allocations inside a layer. None of these controls retain weights
through prefill, add a RAM copy, or change the RAM owner's admission policy.

Numerical validation targets the actual 153 AVX2/FMA/F16C CPU library.
Changing CPU architecture, compiler/libm or quantizer requires rechecking
bitwise equality. UCL/box B timings do not establish 153 speed. See the
experiment report and source/build receipts in `test-vram-tier-assets`.

# DeepSeek-V4.1 GPU-prefill pack source

This is a scheduler/CUDA-adapter feature rather than VRAM-tier residency, but
it uses the same expert pack and direct-I/O machinery. The global runtime
default remains off. The validated `vram21_gpu_chain` deployment profile turns
on all four controls:

```
GGML_SCHED_MOE_COPY_EXPERT_PACK_SOURCE=1
GGML_SCHED_MOE_COPY_EXPERT_PACK_SOURCE_PROMPT_ONLY=1
GGML_SCHED_MOE_COPY_EXPERT_PACK_SOURCE_BATCH=1
GGML_SCHED_MOE_COPY_EXPERT_PACK_SOURCE_CACHE_BYPASS=1
```

At or above `GGML_OP_OFFLOAD_MIN_BATCH` (default 32), selected expert ranges
are coalesced and read with io_uring/O_DIRECT into the existing 16-slot pinned
staging ring before GPU op offload. Prompt-only cache bypass skips both lookup
and admission but leaves the decode cache unchanged. If a direct range cannot
be resolved or read, the scheduler retains the original mmap copy as fallback.
The formal 52-token and 436-token arms took zero fallbacks and produced
bit-identical first-token values and top-10 logits with the switch off and on.

On the 436-token, seven-pass prompt, geometric prefill wall time fell from
216.108 to 74.262 seconds (2.910x). Effective reads were 6.264 GB/s: this is
bounded by the deployment's 7 GB/s H2D envelope, while shaped fio reached
12.26 GB/s, so a faster disk alone is not a useful optimization. Peak file
cache fell 99.49%, from about 15.228 GiB to 0.076-0.079 GiB. `memory.peak`
fell by 5.35 GB (4.99 GiB), from exactly 17,179,869,184 bytes to
11,825,610,752-11,830,030,336 bytes. Decode's geometric point estimate was
-1.97%, but the OFF and ON arm intervals overlap; do not describe that result
as no impact.

Direct reads remove the prefill page-cache constraint, but u-batch remains a
separate bottleneck. Current 100k estimates are about 4.6 hours at u-batch 64
and 37.7 minutes at u-batch 2048. O_DIRECT is necessary, not sufficient, for
the tens-of-minutes target.
# Shared cache-location consumer

`GGML_KIMI_CPU_SINK_VRAM_CACHE_LOCATIONS=1` installs test4's location view on
the same VRAM context. Its default is `0`. `CACHE_COMPUTE=off` uses the original
selected set; changing shrink/bonus/hybrid remains a separate policy decision.
Native queries pin READY slots without training admission. Only the true base
mask trains demand; only final GPU jobs execute, returning unweighted down
vectors. The consumer releases RAM/GPU leases after work and ordered reduction,
then the existing ready-confirmation interface permits duplicate RAM eviction.

The development implementation requires its own real-CUDA and model checks.
The frozen non-provider r4/r5 certificates do not validate this additional path.

# Explicit expert formats (DeepSeek-V4.1 MXFP4)

The tier also exports `kimi_vram_format_get_api` (version 1). The CPU sink uses
it for experts that are not Q3_K SiTU: `prepare` carries the tensor geometry
(weight types, expert count, the RAM cache's layer list) and `launch` the
activation; admit, finish, cancel and the residency ABI are shared with ABI 2.
Slot geometry, role sizes and the input size come from the expert types.

The only engine is MXFP4 up/gate/down with the clamped SwiGLU or SiTU. Preload
`libkimi-mxfp4-parity.so` (built by the `kimi-mxfp4-parity` target from
`kimi-mxfp4-parity.cu` with `--fmad=false --ftz=false`) next to
`libkimi-vram-tier.so`. Its kernel reproduces the AVX2 `ggml_vec_dot_mxfp4_q8_0`
accumulation, and the host bridge between the two matvec stages uses the CPU
sink's activation expressions and the CPU Q8_0 quantizer, so GPU rows are
bitwise equal to CPU `MUL_MAT_ID`. There is no GPU hidden chain for this format.
Checks: `test-kimi-mxfp4-parity <libkimi-mxfp4-parity.so>` (component rows) and
`test-kimi-vram-mxfp4` with both DSOs preloaded (sink end to end).

Runtime parity gate (required). The kernel reproduces exactly one CPU path: the
AVX2 (AVX-VNNI-INT8) `ggml_vec_dot_mxfp4_q8_0` lanes with FMA accumulators and
the `hsum_float_8` order. Any other CPU build or ISA (AVX-512, scalar, ARM,
other compiler flags) computes rows that are close but not bitwise equal. Each
MXFP4 engine therefore checks itself when it is created (first decode token of
a tier context): deterministic splitmix64 MXFP4 rows at both engine column
counts (5120 and 2304 for DeepSeek-V4.1), E8M0 exponents in a realistic band
and at the edges {0, 1, 2, 60, 127, 128, 140, 170}, Q8_0 inputs produced by the
linked CPU `from_float`, four checks of 37 rows x 4 inputs through the
component, compared bit for bit with the linked CPU `vec_dot`. A mismatch fails
closed: creation throws `MXFP4 VRAM parity self-check failed: check=<name>
input=<i> row=<r> cpu=<hex> gpu=<hex> ...`, the tier releases and the sink
returns that error. There is no fallback and no override; only the optional GPU hidden
chain, which makes no bitwise claim, starts without the check (see below). One line reports the
result, e.g. `[kimi_vram_bridge] mxfp4_selfcheck=pass checks=4 jobs=16 rows=592
columns=5120,2304 reference=linked_cpu_vec_dot us=...`. The hidden-row quantizer
needs no gate of its own: the sink and the bridge call the same linked
`from_float`, which holds because ggml-cpu is one shared library linked by both
(the tier cannot link a `GGML_BACKEND_DL` CPU module). `test-kimi-vram-mxfp4`
drives the mismatch path through the test-only export
`kimi_vram_mxfp4_selfcheck_reference_for_testing`.

`GGML_KIMI_CPU_SINK_VRAM_KEEP_ACROSS_REQUESTS=1` (default 0) keeps the arena,
slot heat and residency through prefill instead of releasing them. A tier error
or a decode token of another model/context epoch still releases the arena.

# MXFP4 GPU hidden chain (optional, not bitwise)

`GGML_KIMI_CPU_SINK_VRAM_MXFP4_GPU_HIDDEN=1` (default 0) runs a layer's SWIGLU_CLAMP VRAM
hits as one submission of the component's chain extension (`kimi-mxfp4-chain.cu`, compiled
into `libkimi-mxfp4-parity.so` around the unchanged parity source): one H2D of the job
descriptors and the Q8_0 input row, the up/gate matvec, the clamped SwiGLU with the device
`expf`, device Q8_0 of the hidden rows (max-abs/127, half to even, FP16 scale), the down
matvec on that device buffer and one D2H of the unweighted down rows. The ordered host
reduction is unchanged; SiTU experts keep the bitwise bridge. This path is NOT bitwise equal
to CPU `MUL_MAT_ID`: `test-kimi-mxfp4-chain` reports its error against the bridge (relative
L2 and L-inf of the down rows, equal hidden Q8_0 bytes) and fails above 1e-3 relative L2;
`test-kimi-vram-mxfp4` bounds the sink output the same way. With the switch off nothing
changes: the bridge, its runtime self-check and the parity kernel run as before. With it on
the engine skips the parity self-check and logs `[kimi_vram_bridge] mxfp4_selfcheck=skipped
reason=gpu_hidden_chain`, so the chain also starts on hosts whose CPU `vec_dot` takes another
path; SiTU layers left on the bridge are then not checked for bitwise equality either.
`test-kimi-vram-mxfp4` covers this with a mismatching CPU reference.

`GGML_KIMI_CPU_SINK_VRAM_MXFP4_GPU_KERNEL` selects the chain's matvec: `rows_q8` (default,
one thread per output row), `parity_q8` (the parity lane kernel) or `rows_float` (stage 2
on the float hidden rows without hidden Q8_0; its difference to the bridge is the Q8_0
quantization noise rather than float rounding, so it is only an evaluation option).

`GGML_KIMI_CPU_SINK_VRAM_TIMING=1` (default 0) adds a timing split for both paths, printed at
shutdown: timing events around the copies and kernels of every component submission and the
engine's host times on the tier worker (`[kimi_vram_timing]`), the launch-to-worker handoff
and the sink thread's wait in finish (`[kimi_vram_tier_timing]`).
`.Agent/run-tools/dsv41/vram_timing_split.py` converts them to milliseconds per decode step.
