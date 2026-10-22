# Source provenance

Imported from WiCi AI's internal repository ssd-llama (not public) at commit
`e79cddaa3a76730fa953b969b7f8c19693867fec`; the [import manifest](docs/ssdash/import-manifest.tsv)
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
