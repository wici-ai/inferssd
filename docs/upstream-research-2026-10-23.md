<!-- SPDX-License-Identifier: MIT -->
# Upstream design and model compatibility review, 2026-10-23

## Scope and decision

This second review covers runtime performance designs **and** new-model support,
with llama.cpp as the primary model-adaptation source. Two small candidates have
been implemented on the local `upstream-models-oct23` branch:

1. `8cf056fd62276c51f2ca99bf0e7268d394d4574b`: vocabulary initialization and
   metadata-preview optimization.
2. `1ed86e0a1fbf1c00a602809a91c1709c756d08bf`: Nemotron explicit head-dimension
   compatibility, including synthetic inference regression tests.

No push or PR has been made. The previous `mac-cache-index` candidate remains a
separate branch; this branch does not include that cache change. Separate review
and device qualification are recommended for each concern.

The base is SSDash `f5e6e26ff26d2014cccef0118048039b88163f54`. The llama.cpp scan
covers the latest 60 commits ending at
`f5027fb21307eb264aed483242603d68c87b0fe3`, plus the selected PRs and their patches.
This is an incremental review, not a complete synchronization of all upstream
history since SSDash's imported snapshot.

## Performance candidates

| Source and design | Transfer assessment | Decision |
|---|---|---|
| [llama.cpp #30679](https://github.com/ggml-org/llama.cpp/pull/30679): stop rendering tokenizer arrays after the log preview is filled; reserve vocabulary maps; avoid temporary allocations in merge parsing and byte decoding | Portable host-side changes with no changes to expert routing or tensor arithmetic | Implemented and tested; benefit is vocabulary initialization, not measured generation speed |
| [Strata Metal cache ownership fix](https://github.com/Niko1221/Strata/commit/e6dbbc312a5e4a350e9d29f6d3f9d5e1759547cb) and [CLI follow-up](https://github.com/Niko1221/Strata/commit/b440640929a9): stop lending expert-cache storage to prefill after ownership has moved to the Metal arena | The ownership invariant is useful. SSDash already lets prefill reuse pinned cache bytes without changing admission/heat/recency; misses use transient buffers (`mac-expert-pack-cache.h/.cpp`). This is a different allocator design, not evidence of the same bug | Retain lifetime/pinning checks; do not copy Strata's fixed 1,500-slot cap or loan flags |
| [Strata route metrics](https://github.com/Niko1221/Strata/commit/d276adf27778): distinct experts and new experts per added verification row | Useful measurements for deciding when larger batches actually amortize expert reads. SSDash already has expert access traces, so first derive overlap statistics from those traces | Next experiment: correlate unique expert bytes, hit rate, read time and compute time with token-batch length; no speculative-window code port in this patch |
| [Strata cost-aware draft windows](https://github.com/Niko1221/Strata/commit/687810556040): expected accepted tokens per measured millisecond | Useful decision principle, but tightly coupled to draft acceptance, verification and platform-specific costs. Ordinary prefill ubatches have different constraints | Defer; do not transplant a speculative-decoding policy into prefill or reinstate drafting as a side effect |
| [llama.cpp #30932](https://github.com/ggml-org/llama.cpp/pull/30932): GLM5-next graph/index-input optimization | Depends on the newer GLM5-next graph path; not a generic expert-cache optimization | Defer with the model/graph dependency chain |
| [llama.cpp #30808](https://github.com/ggml-org/llama.cpp/pull/30808): KleidiAI Q8 SVE kernels | Architecture-specific acceleration requiring compatible Arm hardware and kernel qualification | Not an Apple M2 optimization; defer to an appropriate Linux Arm runner |

Strata was refreshed to `5c8dd322528cc84f68f461ad54a506b9da8fa8d4`; the comparison
with the previous inspected revision contains 127 commits. Colibri's inspected
head remains `e0db083cdc9e563e0766cc6fc661ef23c32298e0`. The previous review's
larger design candidates (shared Metal expert arenas, bounded prefetch overlap,
and persisted expert heat) remain research items, not implemented features here.
No Strata or Colibri implementation was copied in this round.

### Implemented vocabulary changes

- Format only enough array elements for the existing 40-character log preview,
  instead of allocating a representation of the entire array. An individual
  element can exceed the preview length; this is not a strict allocation cap.
  Scalar metadata stays complete, including stored adapter metadata.
- Reserve the BPE-rank and token-ID maps before inserting known-size data.
- Parse merges from GGUF strings and move their components into the rank map.
  Preserve the existing Kimi-K2 no-merges branch and first-occurrence behavior.
- Decode GPT-style byte code points with a small lookup table instead of
  allocating UTF-8 strings and hashing them for every byte. Preserve invalid
  byte fallback behavior.

These changes apply to shared host code on macOS, Linux and Windows. Linux,
Windows and CUDA builds were not executed locally; portability is an assessment,
not a claim that those checks passed. No GPU kernel or I/O policy is changed.

## New-model and compatibility candidates

| Upstream change | Missing pieces in this SSDash revision | Decision |
|---|---|---|
| [Nemotron non-standard GQA, #23343](https://github.com/ggml-org/llama.cpp/pull/23343), `0693dbfb915f33233818373b5f6969396658d02c` | Loader assumes Q width equals residual embedding width; converter derives rotary width from hidden size even when `head_dim` is explicit | Ported the dimension changes only; full Domyn support is not claimed |
| Same PR: Domyn-Small tokenizer and Metaspace handling | BPE conversion, byte-fallback types, pre-tokenizer identity, Metaspace metadata and prefix/detokenization behavior must move together | Defer tokenizer port and require reference token-ID parity before advertising the model |
| [Flint MoE, #30928](https://github.com/ggml-org/llama.cpp/pull/30928), `a37cc46104eb50900d5ab4a9ba02a0abdac039db` | Architecture and tensor registry, converter, tokenizer, leading dense layers, per-layer RoPE/SWA, sinks, shared/routed experts and F16 scaling helper; newer graph/model APIs differ | Highest-priority larger MoE port, but not safely completed by registering an architecture name |
| [Nemotron-3(.5)-Embed, #30877](https://github.com/ggml-org/llama.cpp/pull/30877), `52ffa027455662ea59f78d0e4e5a8f38ee63a5fe`, and [tokenizer follow-up #30974](https://github.com/ggml-org/llama.cpp/pull/30974) | Non-causal configuration, pooling and BOS/EOS behavior, multimodal preprocessing and tokenizer fixes; upstream uses modular converters | Defer as an embedding/multimodal package, with reference embeddings and preprocessing tests |
| [MiniCPM-OCR 1.0, #30979](https://github.com/ggml-org/llama.cpp/pull/30979), `a08ab75ae74c36c15accc6c682a989a9e98dbfd2` | The small upstream patch inherits MiniCPM-V 4.6 text/vision paths and projector types absent from this revision | Defer until that stack and image-alignment behavior are ported together |
| [SenseNova U1, #28919](https://github.com/ggml-org/llama.cpp/pull/28919), `7775ffb7e48fa0c7dcffd5815743b3d2fb3b6761` | New text/vision conversion and graph paths, spatial RoPE and non-causal image handling; inspected conversion path rejects MoE configurations | Lower priority for SSDash's expert-streaming scope; not a small MoE speed improvement |

SSDash still uses monolithic conversion and tensor-loading code. Current
llama.cpp model files often assume a newer layout. Cherry-picking a model name
or a conversion alias without its runtime, tokenizer and metadata dependencies
would create misleading support claims.

### Implemented Nemotron dimension fix

For a model with hidden size 4,096, 48 Q heads, explicit `head_dim=128` and
`partial_rotary_factor=0.5`, the previous converter wrote rotary width 42;
the corrected value is 64. The loader now accepts a Q projection width of
`48 * 128 = 6,144` and the corresponding output projection instead of requiring
both attention projections to have width 4,096. K/V retain their GQA dimensions.

The existing conversion fallback without explicit `head_dim` is unchanged.
This is a compatibility fix for an existing architecture, **not** a completed
new model family, full Domyn tokenizer port, or measured MoE performance change.
Full HF conversion and trained-model output parity remain untested.

## Local validation

Hardware: Apple M2, 8 CPU cores, 8 GiB unified RAM; macOS 26.6.2;
AppleClang 21.0.0. Release builds, OpenMP disabled, build parallelism 4.
Synthetic runtime tests use two CPU threads, hidden size 64, vocabulary 32,
one layer, two KV heads, head dimension 16, 4/6 Q heads, partial RoPE width 8,
context 32, batch/ubatch 8, a four-token prompt and one decode token.

| Check | Result and limits |
|---|---|
| Metal + Accelerate build | `llama-server`, vocabulary and Nemotron tests built successfully |
| CPU-only build (Metal, Accelerate, BLAS, CUDA all disabled) | Relevant executables built; 17 CTest entries passed: helper test, 14 tokenizer fixtures, Nemotron runtime, converter metadata |
| Existing tokenizer fixtures with Metal-enabled build | All 14 passed, plus the new helper test |
| Nemotron synthetic runtime, CPU and Metal | Each backend passed 8 valid fixtures and 2 rejected malformed fixtures. Four split/merged comparisons per backend had max logit difference 0 for both prefill and decode output sets |
| Same synthetic runtime test linked against unchanged base | Standard cases passed; non-standard Q width failed to load as expected |
| Converter metadata test | All 5 subcases passed; unchanged base failed 3 explicit-head subcases. The test executes the real class method using isolated base/writer doubles, not a full Torch conversion |
| Adaptive framework | 108 tests passed |
| Mac alias and native contract tests | 86 tests passed with the source package on `PYTHONPATH` |
| Mac launcher/measurement/image checks | 68 passed. Initial sandbox run blocked process inspection and localhost sockets; rerun with those test facilities available passed |

No trained-model inference, Linux/CUDA/Windows device qualification, or full
model HF-to-GGUF conversion was performed. Linux shell-launcher checks require
a Linux environment and remain pending. Synthetic split/merged parity is not
an independent HF numerical oracle. No full-suite or all-platform pass is claimed.

### Vocabulary-only timing

Measured `llama_model_load_from_file(..., vocab_only=true)` using repository
vocabulary fixtures. The baseline is `f5e6e26`; the candidate includes the two
implementation commits above. Each fixture/arm has one excluded warm-up followed
by seven fresh-process samples in alternating arm order. The timed interval
excludes backend initialization and model destruction. Logging output was
suppressed in both arms; metadata formatting still occurs in the loader.
Builds and other test jobs owned by this task had finished before measurement.

| Vocabulary fixture | Baseline median (ms) | Candidate median (ms) | Lower latency |
|---|---:|---:|---:|
| GPT-2 | 46.702 | 30.218 | 35.30% |
| Qwen2 | 132.016 | 77.548 | 41.26% |
| Llama BPE | 160.809 | 93.071 | 42.12% |
| Gemma 4 | 342.073 | 185.907 | 45.65% |

These are local warm-file-cache vocabulary measurements, not model loading,
SSD expert throughput, time to first token, prefill or decode benchmarks.
Seven samples on one machine do not establish a general speedup. Sample values,
fixture hashes and the benchmark harness are retained with the local handoff.

## Reproduce and qualify before merge

Configure a Release CMake build with tests enabled. For the Metal build use
`GGML_METAL=ON`, `GGML_ACCELERATE=ON`, `GGML_CUDA=OFF`, `GGML_OPENMP=OFF`;
for the CPU-only build also disable Metal, Accelerate and BLAS. Build
`test-vocab-loading`, `test-tokenizer-0`, `test-nemotron-gqa` and, for the Metal
build, `llama-server`.

```sh
ctest --test-dir BUILD --output-on-failure \
  -R '^(test-vocab-loading|test-nemotron-gqa|test-nemotron-converter|test-tokenizer-0-)'
BUILD/bin/test-nemotron-gqa --gpu  # fails if a GPU backend is unavailable
python3 tests/test-nemotron-converter.py
python3 -m pytest -q tools/adaptive/tests
PYTHONPATH=tools/ssdash/src python3 -m pytest -q \
  tools/ssdash/tests/test_mac_alias.py tools/ssdash/tests/test_mac_alias_native_contract.py
python3 -m pytest -q tests/test-macos-launcher.py tests/test-macos-measurement.py \
  tests/test-macos-measurement-images.py
```

Before a support claim or merge: run the repository's Linux checks and CUDA
build, confirm tokenizer/real-model output compatibility on a supported trained
checkpoint, and report startup separately from generation performance. For a
future Flint port, test GGUF conversion, tokenizer parity, routing/shared-expert
numerics, pack/alias source support, and dense-to-MoE/SWA transitions before
running the standard 512/4K prompt comparison.

For future cache/batch experiments, record useful expert bytes, physical read
bytes, hits/misses by phase, distinct experts per added token, CPU/GPU time,
resident/peak memory and cold/warm runs. Larger token batches reduce read
transactions only when expert reuse and buffer capacity outweigh the additional
unique experts and memory pressure; this must be measured, not assumed.
