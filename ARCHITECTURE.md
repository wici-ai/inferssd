# InferSSD architecture

The runtime extends the bundled llama.cpp snapshot. Its source attribution is in
[PROVENANCE.md](PROVENANCE.md); upstream integration points are in [PATCHES.md](PATCHES.md).

| Layer | Directory | Owns |
| --- | --- | --- |
| Core | `engine/ggml/src/ggml-ssd/store` | Immutable per-context source indexes and explicit pack/alias parsing |
| Core | `engine/ggml/src/ggml-ssd/io` | Independent syscall and liburing readers; portable pread |
| Core | `engine/ggml/src/ggml-ssd/host` | Process host arbiter, per-context leases, expert cache and page accounting |
| Core | `engine/ggml/src/ggml-ssd/place` | Shared pure placement cost decisions |
| Core | `engine/ggml/src/ggml-ssd/stream` | Expert ranges, dense lookahead, seed ordering and copy observations |
| Core | `engine/ggml/src/ggml-ssd/settings`, `telemetry` | Immutable validated settings, source receipts and owned JSONL writers |
| Executors | `engine/ggml/src/ggml-cpu/ssd`, `engine/ggml/src/ggml-cuda/ssd` | CPU scheduling/compute; CUDA tier, staging, events and format kernels |
| Runtime | `engine/src/ssd` | Context assembly, backend registry binding, model sources, graph admission and diagnostics |
| Other llama features | `engine/src/inferssd` | KV state codecs and positional state serialization |
| Common | `engine/common/inferssd` | CLI glue, model chat parsing, reasoning budgets |
| Server | `engine/tools/server/inferssd` | Prompt cache, checkpoints and request lifecycle |
| Framework | `framework` | Standard-library Python package: source/probe/plan/settings/owned launch |

Core code depends on ggml-base and platform facilities. A CPU executor never
links CUDA; a CUDA executor never links CPU. L3 obtains immutable, size-checked
same-build provider tables through ordinary backend registries and supplies
context-owned services. The scheduler calls a borrowed hook table, so ggml-base
has no dependency on the core implementation. Kernels stay in their backend.

The context owns sources, host registration, writer, transport and diagnostics.
Executors borrow them. Graph failure fences outstanding seed consumers before
reset. A precompute transaction restores touched and purged KV metadata without
allocating; it ends before compute submission. Stateful memory that cannot roll
back a submitted failure clears affected sequences for checkpoint restore or replay.
Backend destruction quiesces transport before destroying CUDA streams.
The writer and source descriptors close after all borrowers stop. Allocation
admission precedes allocation and first touch; committed allocations invalidate
learning intervals when subsequently freed or replaced. Linux uses finite real
cgroup constraints; the Mac cache retains its platform RSS guard.

`inferssd-settings 1` is the production configuration. Retired environment names
are rejected by name without reporting their values. Only `INFERSSD_DEBUG` and
`INFERSSD_TRACE_DIR` are observation channels. Trace/profile require an owned
telemetry writer. Inherited upstream runtime options remain upstream options.
Mac M1/M2 denote project stages; online adaptive Mac M2 is deferred.

Consumed historical launchers and source catalogs remain test fixtures outside
the production module build; unconsumed experiments have been removed from the
current tree. They cannot be selected by the typed production launcher. No
formal benchmark or release claim follows from a module move or syntax check.

Run `python3 scripts/inferssd/check-layering.py` to check these boundaries.

Deployment paths and capabilities are explicit [engine resources](docs/engine-resources.md).
The installed framework needs no source checkout. Native probes belong to
`engine/tools/inferssd-probe`; their execution remains an explicit operation.
