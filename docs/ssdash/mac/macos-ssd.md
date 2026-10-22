> Historical engineering log from the pre-SSDash Mac branch. These results
> have not been revalidated on the current SSDash integration.

# Apple Silicon SSD-backed DeepSeek V4.1 baseline

This profile preserves the working macOS implementation: routed experts and
Engram lookups run on the CPU, while attention, router projections and shared
experts use Metal. RoPE operations with a nonzero rotation offset fall back to
the CPU for correctness. It does not implement a bounded Metal expert cache.

Validated on an M5 Max MacBook Pro with 48 GiB unified memory and macOS 26.4,
using the ten-shard DeepSeek-V4.1-Flash-Q8_0 GGUF. That checkpoint's routed
experts are MXFP4 despite the Q8_0 directory name. Other quantizations and Mac
configurations require separate validation.

## Build and run

Install Apple's command-line developer tools, CMake and Ninja. With Homebrew:

```sh
xcode-select --install
brew install cmake ninja
./scripts/build-macos-metal.sh
```

The build uses Metal and Accelerate, disables CUDA and OpenMP, and produces
`llama-cli`, `llama-server` and `test-backend-ops` in `build-mac/bin`.

Point to the first GGUF shard; keep all shards together. Models are not
downloaded by these scripts.

```sh
export SSD_LLAMA_MODEL=/path/to/DeepSeek-V4.1-Flash-Q8_0-00001-of-00010.gguf
./scripts/run-deepseek-v41-macos.sh chat
./scripts/run-deepseek-v41-macos.sh server
```

The server binds to `127.0.0.1:8080` with one slot. Run one mode at a time to
avoid loading two instances. `SSD_LLAMA_MODEL` is required. The launcher stops if it is unset.

Environment overrides:

| Variable | Default | Meaning |
|---|---|---|
| `SSD_LLAMA_MODEL` | required | GGUF path |
| `SSD_LLAMA_CTX` | `512` | context capacity |
| `SSD_LLAMA_THREADS` | `8` | CPU threads |
| `SSD_LLAMA_PORT` | `8080` | server port |
| `SSD_LLAMA_BUILD_DIR` | `build-mac` under the checkout | build and launcher directory |
| `SSD_LLAMA_BUILD_JOBS` | `8` | parallel build jobs |

Additional llama arguments follow the mode. For the measured larger-prefill
configuration (higher memory requirements, not the conservative default):

```sh
SSD_LLAMA_CTX=8192 ./scripts/run-deepseek-v41-macos.sh server -b 512 -ub 512 -tb 12
```

## Correctness and memory settings

- `LLAMA_MMAP_PREFETCH=0` avoids prefetching the entire oversized model on macOS.
  The loader change is Apple-only and opt-in; the launcher enables it.
- `LLAMA_DSV41_TOPK_REUSE=0` preserves the reference attention path. Reuse changed
  an answer in the earlier 130-token Metal regression case.
- `--cpu-moe`, `--no-op-offload` and the Engram CPU tensor override keep the
  large, streamed weights on the tested CPU path. `-ngl 99` does not mean every
  operation runs on the GPU.
- `--mmap --no-repack --fit off --no-warmup` and `--cache-ram 0` preserve the
  tested loading and runtime policy. Default batch/ubatch remain 32/32.
- Unified memory does not make the roughly 289 GB expert set fit into 48 GiB.
  Removing `--cpu-moe` is not a substitute for a budgeted expert streaming path.

The CUDA-only direct-reader tool is built only on Linux with CUDA enabled.
Server CUDA MoE hooks are guarded by the CUDA build configuration. The independent CLI final-delta change is not included in this integration.

## Validation and retained baseline

The following describes the original branch validation, before integration.
The baseline was built on the target Mac; CLI, streamed and non-streamed server
responses were checked. Checks included a greeting, arithmetic, multi-turn
recall, a batched input verification code and a short explanatory answer.

The 2026-10-14 profiling covered 32–7680 input tokens and batch/ubatch up to
8192. Bigger ubatches improved prefill, but did not improve single-stream decode.
These are measurements of particular inputs and cache states, not performance
guarantees or comprehensive model-quality tests. Some cross-ubatch outputs
differed and still need logits/teacher-forcing investigation.

A subsequent Metal suffix-RoPE experiment passed its operator and basic
functional checks but showed no clear overall speedup and changed some generated
tokens. Its code is excluded from this baseline. The original CPU RoPE fallback
and production defaults are retained. Experimental patches, raw traces and
machine-specific reports are archived outside the repository.

## Next optimization boundary

Keep this branch as the reproducible CPU-expert/Metal-attention starting point.
The next substantial GPU optimization needs selected-expert reads, a bounded
shared-memory cache, expert-ID remapping and synchronization before cache-slot
reuse, using the existing Metal MXFP4 kernels. Validate numerics before measuring
end-to-end speed with fixed continuations.

[Colibri](https://github.com/JustVugg/colibri) is retained as a design comparison
for explicit expert caching and batched reads, not as a measured same-machine
performance baseline. Its model-specific backend and checkpoint requirements
must be checked before comparisons.

This adaptation and its documentation were developed with AI assistance. No
Linux/CUDA hardware regression run was performed as part of the Mac validation.
