> Historical engineering log from the pre-SSDash Mac branch. These results
> have not been revalidated on the current SSDash integration.

# DeepSeek V4.1 expert-pack path on Apple Silicon

This is an opt-in path. The existing `run-deepseek-v41-macos.sh` and `build-mac`
remain the validated baseline. The pack path uses the same Q8 GGUF dense weights
and the MXFP4 routed experts copied without requantization into the Linux v1
expert-pack format. It runs routed expert arithmetic on CPU; attention and
shared experts remain on Metal.

## Data path

For single-token routed `MUL_MAT_ID` calls with 384 experts, the CPU backend obtains each
selected expert slice from the pack index. Multi-token prefill remains on the
validated GGUF mmap path. The reader opens the pack with `F_NOCACHE`,
reads a 4096-byte-padded slice into an aligned buffer, and retains it in a
bounded process-local cache with a per-tensor quota and decaying use counts.
The buffer stays pinned while all ggml worker
threads compute that expert. Other tensor reads still use GGUF mmap. No CUDA,
Linux `io_uring`, or separate GPU VRAM is assumed. The pack is an additional
file: it does not replace the ten GGUF shards.

The pack reader validates the v1 magic, index geometry, slice bounds, tensor
name, expert ID, and exact logical byte length. Structural mismatches stop
inference. The v1 format has no model fingerprint, so a pack from a different
checkpoint with identical tensor names and shapes is not detectable; build it
from the GGUF shards used for inference and verify output tokens against the
original GGUF path. The default cache budget
is 4096 MiB; `SSD_LLAMA_MAC_EXPERT_CACHE_MIB=0` streams without retention.
The cache logs cumulative hit/miss counts, pack bytes read, read wall time,
evictions, and scratch-buffer reuse every 1024 slice requests. Non-retained
expert reads reuse one aligned buffer, avoiding repeated allocation. LFU is
the default admission/eviction policy; `SSD_LLAMA_MAC_EXPERT_CACHE_POLICY=lru`
is available for controlled comparisons. Trace individual expert requests
with `SSD_LLAMA_MAC_EXPERT_TRACE=/path/to/trace.txt` when profiling.

On the 48 GiB MacBook Pro, a 10240 MiB LFU cache improved the measured decode
rate without additional swap growth. The launcher selects 10240 MiB on Macs
with at least 48 GiB unified memory, otherwise 4096 MiB. Override it with
`SSD_LLAMA_MAC_EXPERT_CACHE_MIB`. Cache statistics count bytes returned by pack
reads, not independently measured physical SSD traffic.

## Build and launch

Build both launcher modes into the same configurable build directory:

```sh
./scripts/build-macos-metal.sh
```

The pack builder requires the Python GGUF reader dependencies. Generate the
pack from GGUF shards 1–7 into a filesystem with at least 289 GB free:

```sh
python3 engine/tools/expert-pack/create-moe-expert-pack.py \
  '/path/to/DeepSeek-V4.1-Flash-Q8_0-0000[1-7]-of-00010.gguf' \
  -o /path/to/DeepSeek-V4.1-Flash-Q8_0-MXFP4.expert-pack --align 4096
python3 engine/tools/expert-pack/create-moe-expert-pack.py --inspect \
  -o /path/to/DeepSeek-V4.1-Flash-Q8_0-MXFP4.expert-pack
```

Set both required variables, `SSD_LLAMA_MODEL` and `SSD_LLAMA_MAC_EXPERT_PACK`, then launch
`./scripts/run-deepseek-v41-macos-pack.sh`. This script deliberately uses
`build-mac` (or `SSD_LLAMA_BUILD_DIR`), port 18089 and the same CPU/Metal placement as the baseline.

This first port reuses the Linux pack format and adds an explicit Mac cache.
It does **not** yet include Linux's `io_uring` multi-request pipeline, GPU
expert tier, or CUDA fused kernels. `F_NOCACHE` alone does not make access
faster; the pack path must be checked against the original binary using
identical input token IDs and output tokens, with SSD bytes, cache hits and
memory pressure recorded. Do not promote it to the normal launcher on a
compile-only or small synthetic-pack test.

## Validation on a 48 GiB MacBook Pro

With the same Q8 GGUF and 128 input tokens / 32 output tokens, the unchanged
Mac baseline decoded at 0.973 token/s and the pack path with a 10240 MiB LFU
cache at 2.138 token/s. Prefill was 3.128 and 3.110 token/s respectively.
The pack cache hit 52.43% of 22,320 expert-slice requests and issued 66.54 GB
of logical reads; synchronous `pread` calls occupied 7.12 seconds of the
14.97-second decode phase. Peak process RSS in the 128-token pack run was
29.03 GiB. In the [follow-up five-length comparison](macos-deepseek-v41-benchmark-20261015.md),
output token IDs matched the original at 32, 128, 512, 1024 and 2048 input
tokens. At 1024 and 2048 tokens, pack decode reached 2.072 and 2.080 tok/s,
versus 0.919 and 0.940 tok/s for the original GGUF path; prefill was effectively
unchanged. These are single-stream measurements, not a guarantee for longer or
concurrent workloads.
