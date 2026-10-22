# Muse Spark on Mac: explicit expert cache and compact Metal allocations

This opt-in integration was prepared with AI assistance for the private Mac
deployment. It retains the original binary in `build-mac`; the experimental
binary is built in `build-mac-muse-opt`.

## Changes

- Reuses the existing Mac v1 expert-pack reader, bounded LFU/LRU cache,
  `F_NOCACHE` reads, pinning, and aligned scratch-buffer reuse from
  `source-mac-pack`.
- Extends the single-token CPU `MUL_MAT_ID` pack hook to 256-expert tensors,
  in addition to the previously supported 384-expert tensors. It still requires
  routed-expert tensor names and checks the requested expert's exact byte size
  against the pack index. Pack creation validates Muse's 3D shapes and
  Q4_K/Q5_K/Q8_0 types.
- `LLAMA_MAC_COMPACT_METAL=1` uses allocated Metal buffers for Muse GPU tensors,
  rather than registering the large GGUF span between the first and last
  tensor. Only those GPU tensors are copied. CPU experts remain mapped.
  This is a compact-copy solution, not the upstream disjoint-mapping loader.
- `LLAMA_MAC_ADAPTIVE_PREFILL=1` enables an experimental single-slot server
  controller. It begins at up to 256 tokens, grows within `min(batch, ubatch)`,
  and backs off if the larger full batch's milliseconds/token regress by more
  than 20%. Short checkpoint/tail batches do not train the controller.
  The configured context ubatch capacity is unchanged; submitted prefill
  work sizes vary. Decode is unaffected. This heuristic has no claim of
  finding an optimal size or controlling memory pressure.

The pack hook is decode-only. Multi-token prefill continues to use GGUF mmap.
There is no new asynchronous read pipeline or Metal expert-compute cache.

## Test configuration

```sh
export LLAMA_MMAP_PREFETCH=0
export LLAMA_MAC_COMPACT_METAL=1
export LLAMA_DROP_MMAP_TENSOR_AFTER_UPLOAD=1
export LLAMA_MAC_ADAPTIVE_PREFILL=1
export SSD_LLAMA_MAC_EXPERT_PACK=/path/to/Muse-Q4.expert-pack
export SSD_LLAMA_MAC_EXPERT_CACHE_MIB=8192
export SSD_LLAMA_MAC_EXPERT_CACHE_POLICY=lfu

./build-mac-muse-opt/bin/llama-server \
  -m /path/to/Muse-Spark-Open-816B-A42B-UD-Q4_K_XL-00001-of-00012.gguf \
  --mmap --no-warmup --fit off --no-repack --no-host \
  -ngl 99 --cpu-moe --no-op-offload \
  -c 8192 -b 512 -ub 512 -t 8 -tb 12 --parallel 1 --cache-ram 0 \
  --chat-template-file models/templates/muse-spark.jinja \
  --host 127.0.0.1 --port 18086
```

Use the explicit template: the GGUF's embedded template uses syntax not
supported by this branch's template parser.

Disable the new paths by removing their environment variables. Remove
`SSD_LLAMA_MAC_EXPERT_PACK` for a same-binary pack-off comparison.
Use the original binary for the unmodified baseline.

## Evidence and limits

Run artifacts live outside the source tree in `../muse-opt-20261020`.
The pack contains 47,616 slices (186 tensors × 256 experts), with a total size
of 482,654,191,616 bytes. Its generator records a payload hash and inventory
hash; the reader still uses the original v1 format, which has no embedded
checkpoint fingerprint. Keep the generated inventory and status sidecars with
the pack, and never pair this pack with another model.

Validation checks every index entry and compares 1,674 source/pack byte ranges.
This is sampled readback, not a full file readback checksum.
Selected Q4_K/Q5_K/Q8_0 CPU MUL_MAT_ID tests passed 148/148 before model testing.
CPU, compact-Metal, and Metal+pack short-output checks gate the 512 benchmark.
All three short runs generated the same eight token IDs. Metal pack-off and
pack-on also returned identical reported token/top-5 log probabilities.
The short pack-on run exercised 10,464 pack requests. These checks are useful
regression evidence, not broad model-quality validation.
The complete 512 benchmark results must be read from the run artifacts.

The benchmark uses the supplied ZIP's prompt-512, records the actual model
token count, disables prompt-cache reuse, and generates 256 tokens. It records
prefill and both 128-token windows. As in the historical reporting convention,
the first window uses 128/T128 even though its first token comes from prefill;
the corrected 127-interval rate is also stored.

No system page-cache purge is performed. Model loading, pack creation, and
preceding validation can affect OS cache state. Neither a controlled cold-cache
claim nor an isolated dynamic-ubatch speedup can be inferred from this run.

## Completed prompt-512 run, 2026-10-20

The supplied prompt rendered to 649 input tokens and generated all 256 tokens,
with no prompt-cache reuse and no request error.

| Metric | Previous CPU+mmap run | Integrated Metal+pack run |
| --- | ---: | ---: |
| Prefill tokens/s | 1.900696 | 2.069227 |
| First 128 output tokens/s | 0.114150 | 0.761754 |
| Last 128 output tokens/s | 0.120324 | 0.762593 |
| Request seconds, excluding model load | 2526.604 | 649.545 |

Actual prefill submissions were 256, 389, and 4 tokens. The controller grew
its limit from 256 to 512 after the first full batch; the remaining batches
were tails. This does not demonstrate that 512 is the globally optimal size.

The decode pack reader recorded 379,488 requests, a 21.02% hit rate, and
3,038,075,944,960 logical bytes read. Peak cache allocation was 7,552,892,928
bytes, below the configured 8 GiB budget. Two physical disk samples measured
about 8.7 GB/s on the internal SSD and zero external-disk activity. Swap stayed
at 0.75 MiB during the observed run.

The long generated text differs from the previous CPU run, first at character
146. The two runs therefore have different later routing workloads. The ratios
above are observed end-to-end improvements, not a fixed-continuation A/B result
or a proof of numerical equivalence across CPU/Metal/batch sizes.
The short same-binary Metal pack-off/on test did have identical reported
tokens and probabilities.

Historical upstream llama.cpp's prompt-512 result was 1.937996 prefill,
0.158610 first-half decode, and 0.153657 second-half decode tokens/s.
That run had 666 input tokens, a different loader/template, and uncontrolled
cache state. It is historical context, not a strict matched comparator.

The portable [benchmark summary](benchmarks/muse-mac-512-20261020.json)
contains the measured windows and validation outcomes. Raw events, server logs,
configuration, and comparison files remain in the test machine's
`../muse-opt-20261020/benchmark512/prompt-512/` and
`../muse-opt-20261020/comparison.json`; model weights and generated packs are
not part of the source repository.
The test server exited after completion. The full pack consumes about
449.5 GiB of additional storage; roughly 24 GiB remained on the internal disk.
