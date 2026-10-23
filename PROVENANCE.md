# Source provenance

Imported from WiCi AI's internal repository ssd-llama (not public) at commit
`e79cddaa3a76730fa953b969b7f8c19693867fec`; the [import manifest](docs/inferssd/import-manifest.tsv)
lists the origin of every changed path. Historical source hashes are plain text because that repository is
not publicly accessible.

## Import structure

1. An upstream snapshot of ggml-org/llama.cpp `f42e29fdf199481effd31f281bf095ec6067757b`, without upstream
   project-process files. The `upstream` branch identifies this snapshot.
2. Files marked `keep` or `carry-p6` copied byte for byte from the source seed.
3. The planned CPU-only build guard, excluded launcher-test registration, required golden launcher fixture,
   and documentation cleanup. Runtime algorithms are unchanged by this step.
4. Repository metadata, attribution, community documentation and verification tools.

The golden launcher fixture is copied byte for byte from ssd-llama `1f174b4`. The example at
`examples/simple/simple.cpp` retains its upstream version. Future ports and source-layout changes will be
recorded here by their public pull-request numbers.

## Module origins

| Component | Source |
|---|---|
| Base runtime, ggml and inherited backends | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), MIT |
| DeepSeek V4 support | [nisparks/llama.cpp](https://github.com/nisparks/llama.cpp), Nicholas Sparks; MIT |
| Lightning indexer | Stanislaw Szymczyk, [llama.cpp PR 21149](https://github.com/ggml-org/llama.cpp/pull/21149), via cchuter/llama.cpp |
| DeepSeek V4.1 runtime | [vcruz305/llama.cpp](https://github.com/vcruz305/llama.cpp), `5210c7c5ed61`; MIT |
| Muse Spark model | ggml-org/llama.cpp `ba3c05f71d`; MIT |
| Engram test reference | [antirez/ds4](https://github.com/antirez/ds4), `8db1d1d155cb`; MIT |
| Float reference routines | ARM optimized-routines, glibc and fdlibm; see NOTICE and component licenses |
| Expert streaming and adaptive framework | jiayangx and other WiCi AI developers |

The source repository also incorporated ik_llama.cpp history. Its copyright line is retained to acknowledge
that lineage; the imported-path audit did not identify surviving lines authored by that project's contributors.
The colibri project informed scheduling ideas, but its implementation is not copied into this repository.
Retained literary prompt excerpts are from public-domain works, including Moby-Dick and Pride and Prejudice.

Full third-party copyright and permission notices are in [NOTICE](NOTICE). Existing upstream attribution in
`AUTHORS`, `licenses/` and vendored directories remains in place.

## Prefill and Mac integration

The grouped prefill reservation changes originate in the internal prefill-room
commits `d3b009f705` and `4b70845959`. Unified placement and cost accounting come
from the lane integration ending at `bdd1530696`, followed by the partial base
admission fix `6d8cb2b`. They were applied as reviewed net patches against the
seed; their original repository history was not merged. The standalone earlier
cost patch `ca8d4ba` is superseded by the lane accounting implementation. This
integration is reviewed in [PR #3](https://github.com/wici-ai/inferssd/pull/3).

The initial Apple Silicon cache, launch scripts and historical benchmark assets
come from ustc-sunny's Mac branch at `aee25da468cbdbb7c72d2293f86262d6933da657`.
Unchanged imported files retain that author's GitHub noreply attribution. The
integration adapts the CPU source hook to the current engine, keeps one Muse
parser, preserves the Linux streaming paths, and adds portable cache validation.
The historical benchmark documents describe that original branch, not a measured
performance claim for this InferSSD revision. The newer GGUF alias reader and I/O
pipeline are separate work and require their own model validation. The original
Mac integration is reviewed in [PR #10](https://github.com/wici-ai/inferssd/pull/10),
which supersedes the earlier PR #4.

The GGUF alias source, bounded read-ahead, aligned reads and prefill cache isolation
originate in optimization commits `301cf6f`, `0824683`, `b513c8a` and `c7f76a2`.
The checked cache-budget conversion comes from `709326e`.
They are applied as separate patches to the new history. The CPU hooks preserve
the new engine's Apple-only path; the portable testing guard and prior malformed
pack checks remain. The alias-index generator is included as part of this feature.
Default changes and performance claims require validation of this integrated build.
The Apple mapping-advice correction from `9f00b1ab` is included independently;
its synthetic test verifies the advice path, not a measured memory reduction.
