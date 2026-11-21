# Test gates

Hosted CI runs the native tests labelled `ci`. On Linux, macOS and Windows it builds the target
`inferssd-ci-tests` and runs `ctest -L ci`, which selects the `ci` tests that OS registers. On Linux the
tests run in a systemd scope with an 8 GiB memory limit, which gives the RAM cache cases the finite cgroup v2
limit they need. A test joins by calling `inferssd_ci_test(<executable target> <test>...)` next to its
`add_test` in [engine/tests/CMakeLists.txt](../engine/tests/CMakeLists.txt); the workflow names no tests.
A missing executable, a skipped test or zero selected tests fails the job.

Left out of `ci`: tests that need a GPU, a model or the network; the store fixtures under
`engine/tests/ssd/fixtures/c0-store` that link the test-only legacy runtime; and the bundled llama.cpp tests
other than `test-reasoning-budget` and `test-kv-cache-grow`, which cover InferSSD changes. Where a hosted
runner cannot run a test, CMake says so next to it: `graph_lane` counts only on macOS, the one runner with a
GPU for the lane, and `test-kv-cache-grow`, `ssd_cpu_sink_cache_load` and `ssd_cpu_sink_work_reclaim` do not
count on macOS.

The `unit` CTest label contains only model-free tests that require neither a GPU
nor cgroup setup. Build the targets before running `ctest -L unit`; a missing
executable, skipped test or zero selected tests is not a passing gate. The CUDA workflow compiles the
server for `sm_120a` and `sm_121a` on `main` and on pull requests that change the
engine or build configuration; it does not establish GPU runtime correctness.
Device runs must record both the selected and actually executed tests.

The generic SSD tests use `test-ssd-*` for source files and executables, and
`ssd_*` or the executable name for CTest. The SSD runtime's symbols, log prefixes
(such as `[ssd_cpu_sink_cache]`) and CMake option (`GGML_CUDA_SSD_VRAM_TIER`) use
`ssd` as well. The former `kimi` names came from the predecessor repository, whose
runtime was first written for a Kimi model. Kimi model support and
`test-kimi-deepseek2-guards` keep their model-specific names. Frozen
oracles, the recorded c0-store references and release-comparison sources retain
their original names, paths and hashes, and the retired `GGML_KIMI_*` environment
variables keep their names where the engine and the launcher refuse them.

The CPU sink suite runs on Linux, macOS and Windows against the same ggml
reference graphs, bit for bit. It exercises direct reads, the OS read ring
(io_uring on Linux, IoRing on Windows), affinity, the host memory envelope and
experimental scheduling paths; on Linux also cgroup memory limits. It is not
interchangeable with the portable unit gate. Do not interpret a full-suite run on a small hosted runner as
a production device acceptance test. GPU suites need dedicated launchers
and explicit resource prerequisites before they can become automated gates.

Some bundled upstream tests in a full run depend on outside resources: `test-download-model`
fetches the model that `test-thread-safety` and `test-state-restore-fragmented` use,
`test-tokenizers-ggml-vocabs` clones vocabularies with Git LFS, `test-quant-type-selection`
compares its recorded snapshots with the current Hugging Face files, and `test-jinja-py`
needs `python3` with `jinja2`; exclude them with `ctest -E` where these are unavailable or have changed.

## Q2_0 support

`test-q2_0` checks the upstream type-42 group-64 layout (18 bytes per block),
all packed byte values, independent decoded dot products, 640-column rows and
invalid scales. It runs in hosted CI. The `ssd_cpu_sink_swiglu_q2` and
`ssd_cpu_sink_swiglu_q2_ordinary` tests include mixed
IQ3_XXS up/gate and Q2_0 down experts at 2560/640 geometry, compared bit for bit
with a resident ggml graph for decode and grouped prefill. `ssd_source_index`
checks the corresponding unaligned 180-byte expert aliases and direct reads, and
a two-shard GGUF whose experts it indexes from the headers and reads back.

CUDA supports Q2_0 conversion, row gathering, matrix-vector and matrix-matrix
products, including the SSD expert tier. On an owned GPU, run
`test-ssd-vram-mm`, `test-ssd-vram-generic` and `test-ssd-vram-group`.
These check literal packed codes (including the fourth level), GPU products
against CPU and independent reference products, mixed IQ3_XXS/Q2_0 experts
at 2560/640 geometry, resident cache hits, grouped prefill and layer format
changes. Rows of 640 values use GPU matrix-vector or dequantization plus
cuBLAS products because they do not fill the integer matrix kernel's tile.
`test-backend-ops test -b CUDA0 -p q2_0` checks the standard backend operations.

Synthetic tests do not establish full-model generation correctness: run a model
with Q2_0 experts at the normal VRAM budget and check GPU expert activity and
long-prompt prefill; `--set vram.budget=0` keeps every expert on the CPU for
comparison.

## Shared expert store

`LLAMA_SSD_STORE` is on by default on Linux, macOS and Windows. The store is a
static library that only llama links; each context owns its source index. The
supported configuration uses bundled ggml, shared or static libraries, the CPU
backend, and linked backends (`GGML_BACKEND_DL=OFF`, `GGML_CPU_ALL_VARIANTS=OFF`).
The GPU SSD build (`GGML_CUDA_SSD_VRAM_TIER=ON`, set by the `linux-cuda` and
`linux-rocm` presets) needs no liburing either: on Linux the read ring submits io_uring
requests through system calls. System-ggml and runtime backend-plugin variants remain
outside this store's supported build boundary. Every OS runs configured experts
through the store and the CPU sink. The OFF configuration is retained for explicit legacy
comparison, not as a runtime fallback. `ssd_source_index` checks the store end to
end on every OS.

The legacy store fixtures below are Linux-only. `test-c0-store` and `test-c0-config` link the actual production store target and
compare it with a fixed test reference. `test-c0-parser-failures` injects
allocation and callback failures into the parser to verify handle cleanup.
Their CTest names `c0-store`, `c0-config` and `c0-parser-failures` carry `unit`
and `ci`. The cross-DSO/mock-provider, source-open-hook and direct
I/O fixtures carry `device`; they do not silently join the hosted unit gate.
The production store CMake target does not read the fixture's reference includes
or execute Python. The fixed reference is copied only in the test configuration.

Successful shared initialization emits one receipt:
`[ssd_store] provider=cuda|portable sources=N entries=N published=1`.
Together with loaded-library hashes and direct-file descriptor identity, this
records which provider populated the single owner. It is not a replacement for
the finite-RAM same-sink numerical comparison or normal reader-shutdown checks.

## CUDA handle allocation recovery

On Linux shared CUDA builds, `test-ssd-cublas-retry` links the backend's real
cuBLAS creation/reclaim helper and interposes `cublasCreate_v2` only inside the
test executable. Eight deterministic cases cover success, repeated allocation
failure followed by recovery, absent/busy/deferred reclaimers, zero progress,
callback failure and non-allocation errors. Failed creates never publish a handle.
The fixture does not exhaust device memory; it still selects a CUDA device, so
run it explicitly on an owned GPU alongside `test-ssd-vram-group` and
`test-ssd-vram-elastic`:

```sh
cmake --build build-linux-cuda --target test-ssd-cublas-retry
build-linux-cuda/bin/test-ssd-cublas-retry
```

On an NVIDIA GPU (device 0), the shared-library fixture below checks when a real
graph first creates its cuBLAS handle. It covers a custom matvec, a failed
reservation with no reclaim progress, a successful retry and handle reuse.

```sh
cmake --build build-linux-cuda --target test-ssd-cublas-reserve
build-linux-cuda/bin/test-ssd-cublas-reserve
```

## Device failures

Device failures are not allowlisted. Record each one with its full test name,
reproduction conditions and outcome on the current hardware. A test with a
hardware-specific numeric expectation must document that expectation rather
than become a generic skip.

## Resource prerequisites

The registered sink/prefill I/O fixtures carry `device`; cache-admission cases
additionally carry `cgroup` on Linux and `envelope` on macOS and Windows, where
the native host ledger always has one. The metadata-only prefill graph and cache
compute-policy tests carry `unit`. Modes that need a Linux facility (cgroup v2
limits, `/proc`, forked `RLIMIT_MEMLOCK` children, pread interposition) return
77 elsewhere; a registered case with such a part prints `status=SKIP` for that
part and runs the rest. Existing manual CUDA workloads remain manual; adding
labels does not authorize a device run.

A fixture returns 77 with a reason for insufficient CPU affinity (compute and I/O
participants are reported separately), unavailable io_uring setup, unsupported
aligned direct reads on its actual input filesystem, or a missing required finite
cgroup on Linux. An unreadable native envelope on macOS or Windows is a failure.
The finite-cgroup probe does not infer a payload budget: fixtures may change
reserves or intentionally exercise pressure. Their explicit size and owned-scope
rules remain authoritative, including the load-room minimum check. Use the
intended storage directory through `TMPDIR` (`TMP` on Windows); probing another
mount does not establish that this fixture's filesystem supports direct I/O.

Fixed-buffer tests use the actual `IORING_REGISTER_BUFFERS` result, including the
real ring geometry and concurrently live rings. EPERM or ENOMEM there records a
skip and the current `RLIMIT_MEMLOCK`; a tiny successful registration is not
proof of the full pin budget. Negative memlock tests still deliberately lower
the child limit and assert rejection. Malformed settings, missing files, short
reads, ordinary allocation errors, registration EINVAL/EFAULT, and numerical
mismatches remain failures. No runtime fallback is added.

CTest preserves 77 as skipped in JUnit. The device gate compares every status with
an explicit reviewed baseline and rejects an all-skipped selection. ASan cannot
establish the prefill cache's process-RSS release assertion: that lifecycle test
executes its other assertions but returns 77, requiring a separate normal-build
pass for physical-release acceptance. Device allowlists and hosted runner
capabilities must be measured on the current hardware, not inferred from
historical failure totals; none is recorded in this repository yet.

## Apple Silicon integration

macOS runs configured experts on the same CPU sink and store as Linux and
Windows, so the sink, prefill, store, read-ring and platform suites above are its
expert-path tests. `test-llama-mmap-dontneed` separately verifies Apple's mapping
advice with a tiny temporary file; production memory/performance effects still
require model runs.

The Mac CI build enables Metal and runs a hosted-runner subset of those suites:
the sink at one to three ggml threads, its SITU, SWIGLU_CLAMP and SWIGLU activations,
Muse Spark's experts with their post norm and grouped prefill, bit for bit against
ggml reference graphs, the prefill llama graph, and the store's source index, read ring and platform layer. It also
exercises Muse rendering with `test-chat --template muse-spark` and checks the
Q4_K/Q5_K/Q8_0 CPU `MUL_MAT_ID` cases. This is a compilation and synthetic
correctness gate. The separate target-device gate still needs matched model token
sequences and memory/performance measurements; no model result follows from
hosted CI.

The unfiltered `test-chat` suite is outside the supported Muse rendering gate.
It is not declared passing or silently allowlisted.

## Windows

The Windows CI job runs the framework tests, then builds the engine as
`install.ps1` does (LLVM clang with the Visual Studio C++ tools' environment,
without a GPU backend) and runs the synthetic and platform tests of the Linux
job that Windows builds, the sink and grouped prefill tests of the Mac job, and
the Windows host budget, platform file and both read ring backends. A test that
builds llama's internal classes stays off Windows, because `llama.dll` does not
export them. Like the other hosted jobs, it is a compilation and synthetic
correctness gate.

The CUDA workflow also builds `windows-cuda` (LLVM clang for the C++, `nvcc` with
the Visual Studio host compiler) from NVIDIA's redistributable toolkit archives.
It compiles and links only; no GPU runs it.
