# C0 store extraction experiment

This private experiment demonstrates a narrow extraction boundary. It is not the
C0 production change, F is not frozen, and it does not establish model parity.
This branch also contains an explicitly opt-in linked prototype. Normal builds
are unchanged. `LLAMA_C0_LINKED_STORE_PROTOTYPE=ON` is accepted only for Linux
CPU-only builds; CUDA combinations fail configuration until both consumers can
share one index. It binds the one mandatory sink resolver directly to the store
without `LD_PRELOAD`. This is not production C0 or a frozen F-prime.

The portable index now lives at `ggml/src/ggml-ssd/store/legacy-index.h`; the
649-check oracle includes that same implementation. A separate configuration
fixture checks legacy single-path versus colon-list, overlay order, IO-backend
and first-character boolean semantics. The linked adapter preserves the per-lookup runtime-disable switch, rejects active
lossless transport, and owns a process-fixed
immutable index and borrows its O_DIRECT descriptors to the unchanged sink.
The existing two-stage alignment, io_uring, ledger/cache and arithmetic stay in
place. Unsupported colibri source configuration is explicitly refused. The raw resolver
skips the test-only identity snapshot, retaining the old fcntl check without
adding a per-expert fstat syscall.

Example build (prototype only):

```sh
cmake -S . -B build-c0 -DGGML_CUDA=OFF -DGGML_NATIVE=OFF \
  -DLLAMA_C0_LINKED_STORE_PROTOTYPE=ON -DLLAMA_BUILD_WEBUI=OFF
cmake --build build-c0 --target llama-server --parallel 4
```

The historical preload adapter below remains only a comparison fixture; it is
not used by the linked candidate. Pack and alias initialization now preserves
the full v1 legacy base/overlay/list/page-cache order rather than that adapter's
simplified two-variable setup. Source parse bodies remain mechanically extracted
from the unchanged CUDA file at configure time. GPU transport still uses its old
index; integrating it into this common ownership is future C0 work after F
freezes. Keeping this switch CPU-only prevents two owners from being mistaken
for a completed shared-store cut.

## Run

From the repository root:

```sh
cmake -S experiments/c0-store -B /tmp/ssdash-c0-store-build -DC0_SANITIZERS=ON
cmake --build /tmp/ssdash-c0-store-build --parallel 2
ctest --test-dir /tmp/ssdash-c0-store-build --output-on-failure
```

The experiment needs C++17, POSIX file handles, and Python 3. It has no ggml,
CUDA, Metal, liburing, or model-weight build dependency. It has been run on
macOS arm64 with AppleClang 21, AddressSanitizer and UndefinedBehaviorSanitizer:
649 checks passed. On Linux arm64 (GCC 13.3), the same 649-check test passed
with ASan/UBSan, and a separate real O_DIRECT range/lifetime test passed.

## Extraction and comparison

`extract.py` obtains the current production bodies from
`ggml/src/ggml-cuda/moe_stream_batch_common.cu` at configure time. The source file
and each selected fragment are hashed in the generated `extraction.json`.

The original oracle compiles those free functions with a minimal state object.
The new store compiles the same bodies as members of a builder, by removing only
`static` from the declaration lines. Sorting and duplicate replacement use the
original loops, with the original resolved replacement flag passed explicitly.
The completed index owns its source handles and exposes only const lookup.
These are extraction and ownership changes, not a new parser implementation.
The entire CUDA initializer is not compiled into the oracle: unrelated settings,
statistics, transport workers and unsupported source families are outside this
experiment. A full production F/F-prime comparison therefore remains necessary.

Tests compare success/failure, all index fields, source ordering, file identity,
and logical range results. Independent fixture expectations verify:

- Keys include tensor name, expert ID and logical byte count.
- Duplicate pack keys fail unless replacement is requested; a later pack wins.
- Page-cache packs force replacement and are loaded after aliases.
- A pack key suppresses the same alias key. The first alias wins across TSVs,
  including when pack replacement is enabled. Matching alias source path strings
  reuse a single source handle across TSVs.
- `expert`/`expert_idx`, `offset`/`model_offset`/`offset_bytes`, reordered columns,
  CRLF and nonaligned logical offsets retain the original behavior.
- Legacy numeric prefixes such as `64tail`, the 127-character v1 name truncation,
  and the v1 index starting at byte 40 even when header_size is larger remain
  unchanged. This work does not tighten malformed-input handling.
- Bad headers, a short index, bad alias rows and duplicate failure release owned
  handles. Logical EOF/overflow checks occur at lookup rather than changing what
  the legacy loader accepts.
- Handles survive builder destruction. Replacing the path does not retarget an
  existing index's descriptor; a new index sees the new inode. Destroying an
  index closes its handles and leaves other indexes intact.
- A generated two-layer, 16-expert, three-role corpus contains 96 exact ranges.
  These ranges are format fixtures, not real quantized tensors or model goldens.

Both parsers share extracted production bodies, so the differential check is
specifically a guard on extraction, state ownership, finalization, and lookup.
It is not an independent implementation audit of the legacy parser itself.

## Smallest production cut after F freezes

| Current dependency | C0 treatment |
|---|---|
| pack v1 header and 152-byte entry table | Move the parsing body unchanged into store |
| alias TSV parser and multi-file source catalog | Move unchanged, retaining column names and priority |
| name/expert/logical-size lookup and bounds | Move unchanged; preserve logical offsets |
| `llama-cpu-sink.h` raw-direct resolver lookup | Link the common resolver directly from store |
| buffered/direct file handles | One store/source owner; borrowed handles outlive all consumers |
| `moe-adapter-dispatch.cu` model-shape choice | Keep the existing CUDA dispatch and K3 refusal until C1 |
| CUDA transport, mmap, io_uring and telemetry | Stay in place and consume the shared index |
| CPU bounce alignment, host cache, lane and ledger | No change in C0 |
| colibri, lossless, v2, K3-only source formats | No parser redesign; keep existing paths until their planned C1 deletion |
| broad API injection and other dlsym removal | C5, not this experiment |

The legacy raw-direct wrapper requires a valid Linux O_DIRECT descriptor. This
prototype's borrowed buffered handle only tests lifetime and inode identity; it
is not a replacement for that raw-direct contract. Keep the direct requirement
and source-file range checks in the production adapter. Pack v1 has no type in
its entries; the caller must supply the tensor's type/shape rather than infer it
from byte length. Existing file paths remain interpreted as they are today,
including relative alias paths; do not silently change them to TSV-relative.

Static inspection found one hard CUDA-provided symbol in the Linux sink startup:
`ggml_cuda_moe_expert_pack_resolve_raw_direct_shape`. CPU executor, activation,
room and prefetch symbols are already provided by ggml-cpu. The CUDA unified
memory query in `kimi-cpu-sink.cpp` is optional and treats an absent provider as
false. VRAM must be disabled for a CPU-only sink check. This is a dependency
finding, not proof that the complete CPU-only model path already runs.

The prototype API is deliberately temporary. Production work must use the
planned result/status boundary and final store interface without changing
legacy parse semantics. Preserve the current source/stat identity consumed by
the RAM cache; the device/inode/size checks here do not replace that richer key.

## Model-golden feasibility and remaining gates

A format fixture cannot replace the planned tiny models. After F freezes:

1. Generate real GGUFs, matching aliases and packs for DeepSeek-V4.1 and Muse,
   a few layers and 16 experts. Dimensions divisible by 256 accommodate Q4_K
   and Q5_K; MXFP4 also fits. Generate fixed-seed weights with format-correct
   quantization, and use a minimal deterministic vocabulary.
2. Muse needs its latent projections, pre/post expert norms, residual scales
   and routing metadata. DeepSeek-V4.1 cannot omit Engram entirely: the loader
   requires at least one Engram layer and matching token-map/hash arrays.
   Its hyper-connection graph asserts a count of four. A small legal Engram
   table and valid compressor/index metadata must be generated explicitly.
   Omitting these would test a different graph or merely fail at loading.
3. On Linux, disable the VRAM tier and use a finite memory.max that admits some
   expert cache slots. Confirm counters exercise CPU decode, grouped prefill,
   cache hit/miss and the ledger. An unlimited host envelope does not cover RAM
   cache admission. Keep GPU graph routing out of this CPU-only comparison.
4. Run F's CUDA build with GPU expert execution disabled and compare it with
   F-prime's CPU path on the same machine. Repeat F-prime twice first to test
   the assumed fixed accumulation order, then require token IDs and per-step
   logit checksums to match exactly. Cross-ISA bitwise equality is not assumed.
5. Record the trusted CPU golden on F-prime. Record CUDA golden data on F with
   top-1 and per-type tolerances. Run Mac goldens against F's Mac path separately.

The follow-up below exercises the Linux sink but does not create a golden or
perform a CUDA parity gate. If another hard CPU-only dependency appears, record
it before widening C0; the plan's fallback is CUDA sink checks after each
reorganization batch, not a broad runtime rewrite.

## Linux dependency probe and generated candidates

`make-tiny.py` generates three deterministic candidates, aliases and pack-v1
files. Run with the repository gguf-py package and NumPy available:

```sh
PYTHONPATH=gguf-py python3 experiments/c0-store/make-tiny.py --out /tmp/c0-tiny
```

The files have two layers, 16 experts, and width 256. DeepSeek-V4.1 uses MXFP4;
Muse uses Q4_K and Q5_K. DSV's current sink guards do not accept Q4_K/Q5_K.
MXFP4 uses gguf-py's quantizer. Q4_K/Q5_K use format-valid random code blocks
with explicit finite scales/minima and validate through gguf-py's dequantizer;
this is a synthetic weight generator, not a new quantization algorithm.
Each model has 96 alias/pack entries, all verified byte-for-byte against GGUF.
GGUF sizes are 6807552, 7102464 and 7888896 bytes respectively. The generated
manifest records seeds, hashes and the explicit non-golden status.

`raw-direct-shim.cpp` is a one-process Linux test adapter for the current
mandatory resolver symbol. It supports the explicit pack/alias inputs used by
the probe, with O_DIRECT and range validation. It deliberately does not claim
full legacy initialization/overlay settings support or the complete K3 dispatch
state machine. Its LD_PRELOAD use is confined to this experiment; production C0
still requires direct linkage and shared index ownership.

`probe-tiny.py` uses an existing CPU-only server, starts one exact child PID per
arm and cleans it up. It runs four input tokens and four output tokens, preserving
raw response/log evidence without recording a trusted golden. The existing Muse
post-expert normalization requires READY_FIRST=1. Use only one prefill-threshold
setting, GGML_KIMI_CPU_PREFILL_MAX_TOKENS=16, rather than also setting the legacy
GGML_KIMI_CPU_PREFILL variable.

Observed on a Linux arm64 CPU-only build (GGML_CUDA=OFF, GGML_NATIVE=OFF):

| Candidate | Completion | Whole-lifetime sink calls | Read requests | Read bytes | Read errors / fallback |
|---|---|---:|---:|---:|---:|
| DeepSeek-V4.1 MXFP4 | 4 prompt / 4 output | 10 | 75 | 2764800 | 0 / 0 |
| Muse Q4_K | 4 prompt / 4 output | 10 | 69 | 2543616 | 0 / 0 |
| Muse Q5_K | 4 prompt / 4 output | 10 | 69 | 3108864 | 0 / 0 |

Calls include initialization and grouped prefill; they are not decode-token
counts. All three showed grouped-prefill source receipts and completed with
VRAM disabled. Without the test store adapter, the DSV candidate aborts at the
existing missing-resolver check. Thus the common index cut is sufficient for
these Linux CPU sink paths without compiling CUDA or modifying the server.

The container did not expose memory.max. Even with an explicit 16 MiB cache
budget, capacity and admitted bytes stayed zero. RAM cache admission/hits remain
uncovered, and a finite-cgroup golden gate is still required. DSV's candidate uses
uncompressed sliding attention and a legal small floating-point Engram table;
it does not cover compressed attention or the real 1017 packed Engram format.
No CUDA equivalence, final F/F-prime golden, or performance claim follows from
these dependency probes. All owned server PIDs were verified absent afterward.
