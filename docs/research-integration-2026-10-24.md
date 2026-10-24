<!-- SPDX-License-Identifier: MIT -->
# Research candidate integration, October 24

The `ready-to-pr-version` branch incorporates InferSSD main at
`92b991674f83af430a0228460f3f870944b0fc59`, fetched on October 24, 2026.
Existing research history is preserved; the earlier cache-index candidate is
adapted to the current engine API. This is a review candidate, not an accepted
throughput improvement or a complete new-model qualification.

## Included changes

| Change | Current implementation | Scope |
| --- | --- | --- |
| Vocabulary initialization | `engine/src/llama-vocab.cpp`, `llama-impl.cpp`, `unicode.cpp` and metadata callers | Reserve maps, reduce temporary strings and bound array log previews; scalar metadata stays complete |
| Nemotron dimensions | `engine/convert_hf_to_gguf.py`, `engine/src/llama-model.cpp` | Honor explicit head dimensions and non-standard Q projection width; no Domyn tokenizer port |
| Expert cache indexes | `engine/ggml/src/ggml-cpu/ssd/mac-expert-source.cpp` | Index payload ownership and per-tensor admissions; preserve LFU/LRU decisions, pinning, read-ahead and prefill isolation |
| Research coverage | [Source registry](research-sources.md) | Seven sources including FreeToken; deferred designs remain proposals, not implemented policies |

The cache port uses explicit `cache_options`, preserves the immutable source
index, and retains main's foreign-batch checks, reader-drain behavior and
telemetry failure handling. Non-owning indexes are updated under the existing
mutex and removed before payload recycling. Payload storage is not duplicated.
The global budget eviction scan remains linear in the slot count.

New model and vocabulary tests live under `engine/tests`; cache regressions live
under `engine/tests/ssd`. Research documents now live directly under `docs`.
Third-party attribution is retained in NOTICE and PROVENANCE.md.

## Validation of the combined tree

Tests use synthetic data or existing vocabulary fixtures, not full model weights.
Release builds use AppleClang, four build workers and OpenMP disabled.

| Check | Result |
| --- | --- |
| CPU-only build, with Metal/Accelerate/BLAS/CUDA disabled | Cache source, cache regression, vocabulary, tokenizer and Nemotron targets built |
| Metal + Accelerate build | Same targets and the Mac expert operator built |
| Targeted CTest on each build | 18/18 passed per build: vocabulary helper, 14 tokenizer fixtures, Nemotron runtime, converter and cache |
| Explicit Metal Nemotron inference | 8 valid fixtures and 2 rejected malformed fixtures; all four split/merged comparisons had maximum logit difference 0 |
| Expert source Python driver on CPU-only build | 86 cases passed: source bytes, pipeline, alignment, prefill isolation, budget, bounds, short reads, duplicates and pins |
| Mac expert operator, Metal-enabled build | 24/24 passed: 256/384 experts, 1/32 rows, depth 0/2/4, F32/Q8_0 |
| Framework and integration suite | 2485 passed, no skips, Python 3.9.6 and pytest 8.4.2 with the local framework package installed; 20 warnings from unchanged historical fixtures |
| Production module boundaries and repository layout | Passed |

Reproduction commands (build directories are examples):

```sh
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF \
  -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF -DGGML_CUDA=OFF -DGGML_OPENMP=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_WEBUI=OFF
cmake --build build-cpu --parallel 4 --target test-mac-expert-pack-cache \
  test-mac-expert-source test-vocab-loading test-nemotron-gqa test-tokenizer-0
ctest --test-dir build-cpu --output-on-failure \
  -R 'mac_expert_pack_cache|test-vocab-loading|test-nemotron|test-tokenizer-0'
python3 engine/tests/ssd/test-mac-expert-source.py build-cpu/bin/test-mac-expert-source
python3 -m pytest -q framework/tests tests/integration
```

For Metal, configure a separate build with `GGML_METAL=ON` and
`GGML_ACCELERATE=ON`; also run `test-nemotron-gqa --gpu`. The framework suite needs
a compatible interpreter and an installed package: several transcript fixtures
are bound to Python 3.9 or 3.12, and isolated subprocesses do not inherit
`PYTHONPATH`. An initial Python 3.11/pytest 7.4 run failed those environment
requirements; the compatible isolated run above supersedes it.

## Remaining qualification

The two Linux launcher shell suites were attempted on macOS and failed because
the system Bash lacks associative arrays and Linux tools/cgroup facilities are
absent. They remain pending on Linux. CUDA/Windows qualification, all operator
sizes/thread counts, sanitizers for this integrated tree, trained-model parity,
and full-model 512/4K throughput comparisons were not run in this integration.
Historical benchmark numbers in the earlier reports are not measurements of
this combined revision. FreeToken code has not been ported.
