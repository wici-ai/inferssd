# Baselines: llama.cpp and colibri on four machines

DeepSeek-V4.1-Flash served by the two systems ssd-llama is compared against, on the machines the adaptive
framework is meant for. Both are run as baselines: the only changes are the ones that make them run on a machine,
and nothing is tuned. Measured 2026-10-13 to 10-16. The baseline runs were paused on 2026-10-14 and only the Mac's
were finished afterwards, so some arms are missing (see [Not run yet](#not-run-yet)).

## Summary

First turn, tok/s, prefill / decode (tokens 129-256). "timeout": no answer within the 2700 s an arm gets.

| machine | prompt | ssd-llama | llama.cpp (PR #28696) | colibri |
|---|---|---|---|---|
| DGX Spark | Chinese, 614 tokens | 34.3 / 6.85 | 2.0 / 2.00 | 1.17 / 0.54 |
| DGX Spark | C++, 3,631 tokens | 94.2 / 6.26 | 2.6 / 1.69 | timeout |
| DGX Spark | English doc, 7,993 tokens | 95.8 / 6.16 | not run | timeout |
| 255H + RTX 5090 | Chinese | 23.9 / 3.59 | not run | 0.88 / 0.53 |
| 255H + RTX 5090 | C++ | 74.5-104.2 / 3.10-3.38 | not run | out of memory |
| 255H + RTX 5090 | English doc | 153.2 / 2.09 | not run | out of memory |
| Apple M1, 16 GB | Chinese | (Linux and CUDA only) | 1.01 / 0.19 | 0.31 / 0.24 (174 of 256 tokens) |
| Apple M1, 16 GB | C++ | | timeout (prefill 3,114 of 3,631 tokens) | timeout |
| Apple M1, 16 GB | English doc | | timeout (prefill 2,049 of 7,993 tokens) | timeout |
| 100U + RTX 5060 Ti | all | 19.9-65.8 / 1.80-1.91 | not run (GPU off the bus) | not run |

Where ssd-llama and a baseline both ran, ssd-llama prefills 17-36 times faster than the faster baseline and
decodes 3.4-6.8 times faster. llama.cpp loses on the way it reads the experts: on the Spark, page faults on the
GGUF mmap move about 1 GB/s from an 11 GB/s drive, and every 512-token batch of a prompt reads its experts again.
colibri is slower still on the Spark's ARM CPU, where its expert kernels (AVX only) fall back to scalar code; on
176 its longer prompts run out of the 14 GB the machine has.

## Model files

- **llama.cpp and ssd-llama** load the GGUF `vcruz305/DeepSeek-V4.1-Flash-GGUF`, revision
  `c4a085541cb53f67ee5e57b63d255e80cef286e7`, files `DeepSeek-V4.1-Flash-Q8_0-000NN-of-00010.gguf` (10 shards,
  508 GB). The name says Q8_0, but the file type is `MOSTLY_MXFP4_MOE` (38): the routed experts are MXFP4
  (120 tensors, 288.8 GB; sampled experts are bit-identical to the official FP4 weights), the Engram tables Q8_0
  (209.2 GB), the other weights Q8_0 (6.9 GB), BF16 (2.8 GB) and F32. On every machine each shard's sha256 matches
  the Hugging Face LFS oid (`hfdl.py` checks it; the 176 box's own deployment plan pins the same hashes).
- **colibri** reads only the official safetensors, `deepseek-ai/DeepSeek-V4.1-Flash` revision
  `5254448103b6254fc646dacbca1cbbf20e9216c1`: 48 shards, 510 GB, each checked against its LFS oid (`dl2.py` with
  `dsv41-official-manifest.json`), the small files against `official-small.sha256`. Routed experts FP4, the other
  linear weights FP8 in 32x32 blocks with UE8M0 scales, the rest BF16 (`quantization_config` in `config.json`).
  colibri's Engram sidecar `dsv41_engram.json` comes from its own `tools/prepare_dsv41.py` (sha256 `fb92589b...`,
  the same on every machine).

So the experts are the same numbers in all three systems; the non-expert weights are FP8 in colibri and Q8_0
converted from them in the GGUF.

## Machines

| name | CPU | memory | GPU | drive |
|---|---|---|---|---|
| Spark | DGX Spark GB10, 10 Cortex-X925 + 10 Cortex-A725 | 119.7 GiB LPDDR5X, shared with the GPU | GB10, 48 SM, sm_121 | Samsung Gen5, 11.0 GiB/s sustained; the model on the raw NVMe (the root file system is an overlay on a loop file, 0.5 GB/s) |
| 176 | Core Ultra 7 255H, 6P + 8E + 2 LP-E | 15.2 GiB (13.9 with the box's services stopped) | RTX 5090 32 GB, PCIe Gen4 x4 (6.8 GB/s) | TiPro9000 4 TB Gen5: 12.5 GiB/s cold, throttles when hot (5.3-7.1 GiB/s over 150 s) |
| Mac | Apple M1, 4P + 4E | 16 GB, shared with the GPU | M1, Metal | internal 1 TB SSD |
| 233 | Core 3 100U, 2P (HT) + 4E | 7.3 GiB | RTX 5060 Ti 16 GB, PCIe Gen4 x4 (6.9 GB/s) | TiPro9000 2 TB Gen4, 6.7 GiB/s, no throttling |

176 uses a Core Ultra 7 255H and an RTX 5090. Every arm ran with other services stopped;
a tracing process remained active and held about 600 MiB of GPU memory.

## Systems

### llama.cpp: PR #28696

Upstream llama.cpp has no DeepSeek-V4.1. The baseline is the GGUF author's runtime as proposed upstream:
ggml-org/llama.cpp PR #28696, "model : add DeepSeek V4.1 (DeepseekV41ForCausalLM)", head
`vcruz305/llama.cpp` `feat/deepseek-v41-convert` at `2076412598e99127ea2554467b2b7530856895d3` (open and unchanged
on 2026-10-17). Built from that tree without changes: on Linux
`cmake -B build-cuda -DGGML_CUDA=ON -DLLAMA_OPENSSL=OFF -DCMAKE_BUILD_TYPE=Release` (default CUDA architectures),
on the Mac the default Metal build (`test-backend-ops`: 10112 passed, 0 failed). Before any timed arm a short
greedy question (2+3, the capital of France) must come back right.

The server (`run-llamacpp-pr.sh` on Linux, `mac-arm-pr.sh` on the Mac):

```
llama-server -m <shard 1> -c 131072 -np 1 -t T -tb T -b 2048 -ub 512 --ctx-checkpoints 8 --fit off \
  -ngl 99 -fa on -ot 'ffn_(up|gate|down)_exps=CPU,engram_embd=CPU' \
  --chat-template-file models/templates/deepseek-ai-DeepSeek-V4.1-Flash.jinja --chat-template-kwargs '{"drop_thinking": false}' \
  --no-repack --cache-ram 0 --no-warmup --reasoning off --temp 0 --top-k 1
```

The experts and the Engram tables are read through the GGUF mmap and computed by the CPU backend; everything else
is on the GPU. On Linux the server runs in a systemd scope with `MemoryMax` = MemAvailable - 512 MiB at its start
and no swap, and with the thread count ssd-llama uses on that machine (Spark 10, 176 14); the Mac has no cgroups
and the server gets llama.cpp's default thread count.

Changes needed to run it:

- **One `-ot` for experts and Engram.** The PR places nothing itself, and `-ngl 99` alone asks the GPU for the
  209 GB Engram tables. llama.cpp keeps only the last `-ot`, so both patterns go into one.
- **Spark: `--load-mode mmap`.** On an integrated GPU llama.cpp's automatic mode reads the whole model into memory
  instead of mapping it.
- **Mac: the context of each prompt.** At `-c 131072` and at 16384 the graph reserve runs Metal out of memory
  (`kIOGPUCommandBufferCallbackErrorOutOfMemory`; the process then dies with a bus error): the indexer and mask
  buffers grow with the context. Each prompt got the smallest context that holds its first turn, its length plus
  256 rounded up to 256: 4096 for the Chinese and C++ prompts, 8448 for the English one. Attention covers the used
  cells only, so the context changes the reserve, not the work per token. The second turns of the C++ and English
  prompts would not have fit; both arms timed out in the first turn.

Not run: the "fill the VRAM" variant that also puts the first K expert layers on the GPU (planned K=8 on the Spark
and K=2 on 176; on 233 no expert layer fits next to the dense ones). On the Mac the experts were not put on Metal:
loading maps the whole 508 GB model as Metal buffers, and an earlier attempt with another llama.cpp build froze
the machine until the kernel watchdog restarted it.

### colibri

JustVugg/colibri at `ce370e87d`, its CPU engine `deepseek_v41`, built with `make deepseek_v41 ARCH=native CUDA=0`
(`build-colibri.sh`; on the Mac with Homebrew's libomp), unchanged. Each arm (`colibri-arm.sh`,
`mac-colibri-arm.sh`, both through `colibri-replay.py` and the engine's own DATA protocol) runs with

```
CTX=131072 OMP_NUM_THREADS=T OMP_DYNAMIC=FALSE MALLOC_ARENA_MAX=1 V41_DIRECT=1 V41_READ_DEPTH=8
V41_ENGRAM_ROWS=65536 V41_STATS=1 COLI_KV_PREFIX=0 V41_DSPARK=1 V41_DSPARK_MINACC=60 SEED=61004
```

with T = 10 on the Spark, 14 on 176 and the 8 physical cores on the Mac, and the same memory scope as llama.cpp on
Linux. The per-layer expert cache cap is the one colibri's planner gives on the idle machine with the page cache
dropped (`coli plan --ctx 131072 --json`, `tiers.ram.cache_slots_per_layer`, at least 1; `colibri-prepare.sh`):
81 on the Spark, 0 and so 1 on 176 and on the Mac. The engine decodes greedily, with its DSpark draft speculation.

Changes needed to run it:

- **Mac: Python 3.12.** colibri's `coli` needs Python 3.10 or later (`dataclass(slots=True)`) and macOS has 3.9; its
  venv is built on a standalone CPython 3.12.14.
- **Context fallback.** A prompt whose arm dies without a turn (as opposed to timing out) runs again at ctx 16384.
  This only happened on 176.

### ssd-llama, for reference

- **176 and 233:** main `2db67387e` with the launcher's defaults (176 with `VRAM_RESERVE_MIB=576` for the trace
  server). On 176 the run-time cache sizing of this branch (`3ec7bf8e6`) ran in the same session, alternating with
  main, and came out the same within the drive's throttling noise (equal bytes read).
- **Spark:** with main's defaults the server takes the host's memory at the first decode token: the GPU tier sizes
  itself from MemFree and `cudaMalloc` is not charged to the cgroup; the test watchdog stopped it at 6-8 GB
  MemAvailable. The numbers are from this branch's run-time cache sizing, `3ec7bf8e6` (the same patch as the
  measured build), with the launcher's defaults.
- **Mac:** not run; the engine needs Linux (io_uring, O_DIRECT) and CUDA.

## Protocol

- Two retained real texts with a one-line instruction (`prompts/`): C++, 3,631 tokens (an excerpt
  of `src/models/deepseek41.cpp`, the Engram path), and English documentation, 7,993 tokens.
  These historical inputs will be replaced with reproducible benchmark fixtures before release.
- Every arm starts a new server after dropping the model files from the page cache (Linux: `posix_fadvise
  DONTNEED` on every shard; Mac: `purge`), then two turns: the second sends the same text again after the first
  answer, in the same conversation. 256 tokens per turn, greedy, thinking off. The Linux arms of llama.cpp and
  ssd-llama ignore EOS; colibri has no such option.
- Prefill is prompt tokens over prompt time from the server's timings (colibri: over the time to the first
  token). Decode is the rate of tokens 129-256 from the streamed token times (colibri on the Mac, which stopped at
  174 tokens: tokens 2-174).
- 2700 s per arm, loading included; a turn not done by then is "timeout".
- On Linux every arm starts with a few seconds of fio random reads of the model files, to record the drive's
  state (Spark 11.4-12.5 GiB/s, 176 12.3-13.0 GiB/s).

## Results

tok/s; "/" separates the first and the second turn.

### Spark

| system | prompt | prefill | decode 129-256 | notes |
|---|---|---|---|---|
| ssd-llama, run-time sizing | Chinese | 34.3 / 67.2 | 6.85 / 6.27 | |
| | C++ | 94.2 / 116.7 | 6.26 / 5.82 | |
| | English doc | 95.8 / 85.2 | 6.16 / 5.59 | second server: 95.4 / 72.5, 6.13 / 5.39 |
| llama.cpp | Chinese | 2.0 / 2.0 | 2.00 / 1.84 | time to first token 304 s; prefill read 294 GB from the drive at 0.97 GB/s, decode 0.53 GB per token |
| | C++ | 2.6 / not run | 1.69 / not run | time to first token 1417 s, prefill read 1,329 GB at 0.94 GB/s; stopped in turn 2 when the baselines were paused |
| | English doc | not run | | |
| colibri, cap 81 | Chinese | 1.17 / timeout | 0.54 / timeout | time to first token 526 s; RSS 73.5 GB, expert hit rate 79.7 % |
| | C++ | timeout | | no token within 2700 s |
| | English doc | timeout | | no token within 2700 s |

colibri's 524 s for the Chinese prompt are 264 s of matrix products and 55 s of reads (10.2 GB/s): its expert
kernels (`expert_ffn.h`, `fused_simd.h`, `gsgemv.h`) only have AVX versions.

### 176

| system | prompt | prefill | decode 129-256 | notes |
|---|---|---|---|---|
| ssd-llama, main | Chinese | 23.9 / 23.1 | 3.59 / 2.35 | |
| | C++ | 104.2 / 71.5 and 74.5 / 59.2 | 3.38 / 2.87 and 3.10 / 2.86 | two servers |
| | English doc | 153.2 / 79.8 | 2.09 / 2.54 | |
| llama.cpp | all | not run | | the first arm was stopped within minutes when the baselines were paused |
| colibri, cap 1, ctx 131072 | Chinese | 0.88 / timeout | 0.53 / timeout | time to first token 696 s, RSS 13.3 GB |
| | C++ | killed | | over the 14.1 GB memory limit after 17 min of prefill |
| | English doc | killed | | over the limit 9 s after start |
| colibri, cap 1, ctx 16384 | C++ | timeout | | no token within 2700 s |
| | English doc | invalid | | its scope's limit came out at 5.0 GB (MemAvailable at that moment, against 14.0-14.2 GB in the other arms) and it was killed after 2 s; to be run again |

The drive throttles during long reads, so ssd-llama's prefill moves with its temperature (2.3-7.3 GB/s here);
the 116.7 / 3.72 that main reached on the C++ prompt the day before with a cool drive shows the spread. At
colibri's 0.88 tok/s the C++ prompt alone would take more than 4000 s to prefill.

### Mac

| system | prompt | prefill | decode | notes |
|---|---|---|---|---|
| llama.cpp, ctx 4096 | Chinese | 1.01 / timeout | 0.19 / timeout | time to first token 610 s, loading 172 s; the answer is coherent Chinese |
| llama.cpp, ctx 4096 | C++ | timeout | | 3,114 of 3,631 prompt tokens after 2,503 s (1.24 tok/s) |
| llama.cpp, ctx 8448 | English doc | timeout | | 2,049 of 7,993 prompt tokens after 1,576 s (1.30 tok/s) |
| colibri, cap 1, ctx 131072 | Chinese | 0.31 / timeout | 0.24 (tokens 2-174) | time to first token 1961 s; 174 of 256 tokens when the arm ran out of time; coherent Chinese |
| | C++ | timeout | | no token within 2700 s |
| | English doc | timeout | | no token within 2700 s |

16 GB does not hold the experts a token uses, so both systems page from the SSD for every token. llama.cpp
prefills three times faster; both decode at about 0.2 tok/s.

### 233

No baseline numbers: its RTX 5060 Ti dropped off the PCIe bus on 2026-10-14 and has not been seen since.
ssd-llama main with the launcher's defaults, measured 2026-10-13:

| prompt | prefill | decode 129-256 |
|---|---|---|
| Chinese | 19.9 / 20.8 | 1.91 / 1.85 |
| C++ | 57.5 / 56.8 | 1.83 / 1.72 |
| English doc | 65.8 / 63.6 | 1.80 / 1.68 |

## Not run yet

- Spark: llama.cpp's English prompt and the C++ prompt's second turn; the first-8-expert-layers variant.
- 176: every llama.cpp arm (plain and first 2 expert layers on the GPU); colibri's English prompt at ctx 16384.
- 233: everything, once the GPU is back (llama.cpp plain with 8 threads; colibri after the official download).

## Files

The scripts are the ones the numbers above came from. They expect this directory as `$ROOT/bench` (the prompts in
`$ROOT/bench/prompts`), `colibri-replay.py` also in `$ROOT`, the colibri tree in `$ROOT/colibri` and its venv in
`$ROOT/colibri-venv`; on the Mac `ROOT` is `~/dsv41`.

| file | role |
|---|---|
| `hfdl.py` | the GGUF shards: resumable range download, sha256 of each against the LFS oid |
| `dl2.py`, `dsv41-official-manifest.json`, `official-small.sha256` | the official safetensors and their hashes |
| `session-pr-baselines.sh` | builds the PR, the coherence check, then the plain (and first-K-layers) arms of one Linux machine |
| `run-llamacpp-pr.sh` | the PR's server on Linux: memory scope, `-ot`, context, load mode |
| `arm.sh`, `replay-stream.py` | one cold arm on Linux: drive probe, page cache drop, server, the two turns with per-token times, summary |
| `mac-arm-pr.sh` | the same for the PR on the Mac (`CTX` sets the context) |
| `build-colibri.sh`, `colibri-prepare.sh` | colibri's engine and venv; its Engram sidecar and planner cap |
| `colibri-arm.sh`, `mac-colibri-arm.sh`, `colibri-replay.py` | one colibri arm on Linux / on the Mac |
| `prompts/replay-real-*-cold-hot.json` | the retained texts |

The run directories stay on the machines (`runs/` under each work directory): Spark
`20261014-174552-pr28696-plain-real-code-3p6k-cold-hot`,
`20261014-0{10717,15222,23728}-colibri-cap81-*`; 176 `20261014-{144740,153336,155115}-colibri-cap1-*`,
`20261014-{172557,181102}-colibri-cap1-ctx16384-*`, ssd-llama main `20261014-*-old2-*`; Mac `20261016-*`.
