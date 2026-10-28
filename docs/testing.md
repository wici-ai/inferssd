# Test gates

The `unit` CTest label contains only model-free tests that require neither a GPU
nor cgroup setup. Build the targets before running `ctest -L unit`; a missing
executable, skipped test or zero selected tests is not a passing gate. CI uses an
explicit subset to keep the default check small. The CUDA workflow compiles the
server for `sm_120a` and `sm_121a` on `main` and on pull requests that change the
engine or build configuration; it does not establish GPU runtime correctness.
Device runs must record both the selected and actually executed tests.

The Linux CPU sink suite also exercises io_uring, O_DIRECT, affinity, cgroup memory
limits and experimental scheduling paths. It is not interchangeable with the
portable unit gate. Do not interpret a full-suite run on a small hosted runner as
a production device acceptance test. GPU and cgroup suites need dedicated launchers
and explicit resource prerequisites before they can become automated gates.

## Shared expert store

Linux enables `LLAMA_SSD_STORE` by default. The supported configuration uses
bundled ggml, shared libraries, the CPU backend, and linked backends
(`GGML_BACKEND_DL=OFF`, `GGML_CPU_ALL_VARIANTS=OFF`). The CUDA SSD transport
(`GGML_CUDA_KIMI_VRAM_TIER=ON`, set by the `linux-cuda` preset) additionally
requires liburing; configuration fails without it. CPU-only builds do not need
liburing: the CPU sink submits io_uring requests through system calls. Static,
system-ggml and runtime backend-plugin variants remain outside this store's
supported build boundary. Mac's reader path is unchanged. The OFF configuration
is retained for explicit legacy comparison, not as a runtime fallback.

`test-c0-store` and `test-c0-config` link the actual production store target and
compare it with a fixed test reference. `test-c0-parser-failures` injects
allocation and callback failures into the parser to verify handle cleanup.
Their CTest names `c0-store`, `c0-config` and `c0-parser-failures` carry `unit`
and run in hosted CPU CI. The cross-DSO/mock-provider, source-open-hook and direct
I/O fixtures carry `device`; they do not silently join the hosted unit gate.
The production store CMake target does not read the fixture's reference includes
or execute Python. The fixed reference is copied only in the test configuration.

Successful shared initialization emits one receipt:
`[ssd_store] provider=cuda|portable sources=N entries=N published=1`.
Together with loaded-library hashes and direct-file descriptor identity, this
records which provider populated the single owner. It is not a replacement for
the finite-RAM same-sink numerical comparison or normal reader-shutdown checks.

## Device failures

Device failures are not allowlisted. Record each one with its full test name,
reproduction conditions and outcome on the current hardware. A test with a
hardware-specific numeric expectation, such as MXFP4 parity on GB10, must
document that expectation rather than become a generic skip.

## Linux resource prerequisites

The registered Linux sink/prefill I/O fixtures carry `device`; cache-admission
cases additionally carry `cgroup`. The metadata-only prefill graph and cache
compute-policy tests carry `unit`. Existing manual CUDA workloads remain manual;
adding labels does not authorize a device run.

A fixture returns 77 with a reason for insufficient CPU affinity (compute and I/O
participants are reported separately), unavailable io_uring setup, unsupported
aligned O_DIRECT on its actual input filesystem, or a missing required finite
cgroup. The finite-cgroup probe does not infer a payload budget: fixtures may
change reserves or intentionally exercise pressure. Their explicit size and
owned-scope rules remain authoritative, including the load-room minimum check.
Use the intended storage directory through `TMPDIR`; probing another mount does
not establish that this fixture's filesystem supports direct I/O.

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

`mac_expert_pack_cache` runs on Linux and macOS using a synthetic three-expert
pack. It checks malformed headers and index entries, LFU retention/eviction and
pinning while another expert is acquired. Linux uses the same reader/cache code
without the Apple-only `F_NOCACHE` call. A malformed zero `data_start` is rejected
before unsigned index-geometry arithmetic. ASan/UBSan are useful for this test.
Four concurrent callers also exercise two decode and two prefill batches with a
1 MiB cache, keeping three 1 MiB slices pinned until reverse-order release.

`test-mac-expert-source` and its Python fixture runner cover pack and multi-shard
GGUF alias sources, unaligned and exact-EOF slices, synchronous and asynchronous
short reads, pinned entries, prefill cache isolation and staging backpressure.
The large-budget fixture sets a 24 GiB limit but allocates only small selected
slices. Run it with `python3 engine/tests/ssd/test-mac-expert-source.py build-macos-metal/bin/test-mac-expert-source`.
It is model-free and supports ASan/UBSan and TSan builds. Launcher and metadata
validation tests run with `python3 -m pytest -q tests/integration/test-macos-launcher.py`.

On Apple Silicon, `mac_expert_operator_*` executes the production CPU source hook
for 256/384 experts, 1/32/512 token rows, F32/Q8_0 weights, and read depths 0/2/4.
Each case compares both broadcast and per-route inputs against a resident graph,
then zeroes the resident weights before the alias computation. All output bytes
must match, so a bypassed source hook cannot pass accidentally.
`test-llama-mmap-dontneed` separately verifies Apple's mapping advice with a tiny
temporary file; production memory/performance effects still require model runs.

The Mac CI build enables Metal, runs that portable cache test, exercises Muse
rendering with `test-chat --template muse-spark`, and checks the Q4_K/Q5_K/Q8_0
CPU `MUL_MAT_ID` cases. This is a compilation and synthetic correctness gate.
The separate target-device gate still needs matched model token sequences and
memory/performance measurements; no model result follows from hosted CI.

The unfiltered `test-chat` suite is outside the supported Muse rendering gate.
It is not declared passing or silently allowlisted.
