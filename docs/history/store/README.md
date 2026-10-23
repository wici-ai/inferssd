# Store extraction tests

The production parser and shared catalog are in `ggml/src/ggml-ssd/store`.
This directory holds the frozen legacy reference and model-free format,
configuration, publication, descriptor-lifetime and exception fixtures. It is
not a second production parser. The recorded runtime references below supplement
these fixtures; synthetic parser checks alone do not establish runtime equivalence.

## Build boundaries

On Linux, `LLAMA_SSD_STORE=ON` is the default. It requires the bundled ggml,
`BUILD_SHARED_LIBS=ON`, `GGML_CPU=ON`, `GGML_BACKEND_DL=OFF` and `GGML_CPU_ALL_VARIANTS=OFF`.
CPU-only builds use the portable initializer. CUDA builds must also specify
`-DGGML_CUDA_MOE_STREAM=ON -DGGML_CUDA_MOE_STREAM_BATCH=ON`; configure rejects a
missing provider instead of silently selecting the portable one. The framework
build launcher already supplies these streaming settings. System ggml, static
consumers and runtime-loaded backend plugins are outside this C0 support scope.
The Mac reader path is unchanged and this Linux store defaults off there.

```sh
cmake -S . -B build-c0 -DGGML_CUDA=OFF -DGGML_NATIVE=OFF \
  -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_WEBUI=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-c0 --parallel 2 --target llama-server \
  test-c0-store test-c0-config test-c0-parser-failures
ctest --test-dir build-c0 --output-on-failure --no-tests=error \
  -R '^(c0-store|c0-config|c0-parser-failures)$'
```

`LLAMA_SSD_STORE=OFF` is retained only for explicit legacy comparison. It does
not provide the supported CPU-only sink path. Remove the old prototype cache
option `LLAMA_C0_LINKED_STORE_PROTOTYPE`; it is no longer accepted. Library/server
builds with `LLAMA_BUILD_TESTS=OFF` do not need Python or any files in this
experiment directory. Test configurations use this directory's frozen fixtures.

## Standalone fixtures

The same fixtures run separately on POSIX systems without ggml, CUDA, Metal or
model weights. CMake copies the fixed oracle includes for tests only; Python is
not needed. Neither parser source nor its reference is regenerated from the
other during configure.

```sh
cmake -S experiments/c0-store -B /tmp/ssdash-c0-store-tests -DC0_SANITIZERS=ON
cmake --build /tmp/ssdash-c0-store-tests --parallel 2
ctest --test-dir /tmp/ssdash-c0-store-tests --output-on-failure
```

The ordinary `unit` gate includes `c0-store` (649 checks), `c0-config` and
`c0-parser-failures`. The last fixture injects failures at actual parser
allocations and at a source-open callback, checking descriptor cleanup.
The cross-DSO mock-provider, source-open hook and Linux direct-reader fixtures
carry `device`: their direct-I/O capability depends on the actual fixture
filesystem. A skipped or unexecuted test is not a passing device gate.

The oracle in `oracle/` was frozen before the adapter change; its original source
blob and SHA-256 are recorded in `oracle/extraction.json`. `test.cpp` compiles
those original free functions independently of the production parser. It checks
all entry fields, ordering, failure results, source identity and logical ranges:

- Pack keys include name, expert and byte count. Duplicates fail unless replacement
  is enabled; later pack sources then win.
- Pack keys suppress matching alias keys. The first alias wins across TSV files,
  including when pack replacement is enabled. Matching paths share a source.
- Base, overlay, extra overlay and list order is preserved. Page-cache sources
  load after aliases and force replacement.
- Existing TSV column aliases, CRLF, numeric-prefix parsing, v1 name truncation,
  the byte-40 index start and logical-range checks remain unchanged.
- A borrowed descriptor keeps its original inode across path replacement and
  lives until its owner resets. A failed initialization never publishes a half
  catalog. Duplicate/late registration and retired providers cannot select a
  second owner.

The test reference and production parser share historical origins. This is an
extraction/ownership regression check, not an independent audit of the legacy
format's validation policy. CUDA transport setup, lazy mmap state, preload,
reporting and reader shutdown stay in their transport boundary. See
[SHARED-PROVIDER.md](SHARED-PROVIDER.md) for ownership and unload limits.

`raw-direct-shim.cpp`, `probe-tiny.py` and `probe-linked.py` retain earlier
feasibility fixtures. The preload shim is not used by production. The linked
probe compares resident CPU arithmetic and refuses CUDA libraries; it is not
the actual-provider same-sink F/F-prime gate. Do not reuse its old protocol or
historical resident/sink results as C0 acceptance evidence.

## Recorded runtime references

The Linux aarch64 CPU-only sink was compared with F using three synthetic model
types (MXFP4, Q4_K and Q5_K), each with pack and GGUF-alias sources. All six cases
matched the complete vocabulary logits and selected IDs byte for byte across F
and two independent CPU-only candidate processes. Each process repeated its
request after clearing KV. Grouped prefill, cache admission/reuse, direct reads,
actual loaded images and normal cleanup were also checked under a finite memory
limit with zero swap. See [the CPU reference](reference/cpu-aarch64/README.md).

The separate [GPU32 reference](reference/gpu32-aarch64-sm121a/README.md) records
original F outputs for a 32-token GPU graph prefill followed by CPU-sink decode.
All six original/candidate pairs matched their four full raw rows exactly.
The run required actual BATCH1 io_uring reads, GPU copies and computation,
complete per-batch CQE histories, zero fallback/errors and normal cleanup.
It does not cover GPU decode or a nonzero VRAM tier. The original worker checked
DONE live; the saved artifacts retain its completion receipt, not raw DONE lines.

These references describe their exact compiler and execution profile. They do
not establish cross-architecture equality, large-model quality, or throughput.
The original failed preparation attempts remain in the migration evidence;
the accepted recording uses the engine's existing request-phase notifications.
Later changes must keep numerical comparison separate from provider, source,
path coverage and resource verification.
