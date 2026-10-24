> Historical engineering log from the pre-SSDash Mac branch. These results
> have not been revalidated on the current SSDash integration.

# DeepSeek-V4.1-Flash on a 48 GiB MacBook Pro — 2026-10-15

This report combines the completed 32, 128, 512, 1024, and 2048-token runs. The 1024/2048 runs were added after the earlier short-prompt comparison; the three short cases were **not rerun**. Historical tables can be rendered from the saved [short](../../../docs/history/benchmarks/macos-deepseek-v41-20261015/summary-short.json) and [long](../../../docs/history/benchmarks/macos-deepseek-v41-20261015/summary-long.json) result summaries by [render.py](../../../docs/history/benchmarks/macos-deepseek-v41-20261015/render.py).

## Setup and interpretation

The three Q8 implementations used the same ten-shard DeepSeek-V4.1-Flash Q8 GGUF, exact prompt token IDs, 8192-token context, `-b 512 -ub 512`, CPU routed experts, Metal attention, one fresh server per run, greedy sampling, and 32 generated tokens. The macOS file cache was retained; these are not cold-SSD measurements. The pack implementation used the MXFP4 expert pack built from those GGUF shards, a 10 GiB LFU expert cache, and the same CPU/Metal placement. All cases ran one engine at a time.

The comparator named “llama.cpp” is the runnable `vcruz305/llama.cpp` DeepSeek V4.1 fork, **not** unmodified `ggml-org/llama.cpp`, which rejected this `deepseek41` GGUF in the earlier check. Colibri used the official Safetensors checkpoint, 18 OpenMP threads, 14 expert-cache slots per layer, CPU execution, and disabled DSpark drafts. It has no equivalent llama.cpp batch/ubatch flag. Its output token IDs differ from Q8, so the Colibri percentages are whole-engine observations across different weight formats and computation paths. The Q8 fork also diverged from the original at output token 3; original and pack produced identical 32-token outputs at every measured length.

Each cell gives tok/s and `(engine rate / fork rate − 1) × 100%` for the same prompt length. 🟩 marks the highest observed rate. Percentages use unrounded rates. The Q8 prefill and decode rates come from server timing fields; Colibri rates are token count divided by its logged phase duration. Startup/model load is excluded. At 32 and 128 tokens, original and pack rates pool two interleaved runs; all other Q8 results and each Colibri result are single runs. Differences smaller than about 1% in prefill should not be attributed to pack/cache.

## Prefill

| Input tokens | llama.cpp V4.1 fork (baseline) | Original ssd-llama | Pack + cache | Colibri |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 1.305 (0%) | 1.609 (+23.3%) | 1.584 (+21.4%) | 🟩 **1.766 (+35.3%)** |
| 128 | 2.727 (0%) | 3.122 (+14.5%) | 🟩 **3.144 (+15.3%)** | 1.944 (−28.7%) |
| 512 | 7.410 (0%) | 🟩 **8.301 (+12.0%)** | 8.252 (+11.4%) | 2.001 (−73.0%) |
| 1024 | 8.261 (0%) | 🟩 **8.984 (+8.7%)** | 8.949 (+8.3%) | 1.748 (−78.8%) |
| 2048 | 8.871 (0%) | 🟩 **9.347 (+5.4%)** | 9.336 (+5.2%) | 1.463 (−83.5%) |

## Decode

| Input tokens | llama.cpp V4.1 fork (baseline) | Original ssd-llama | Pack + cache | Colibri |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 1.217 (0%) | 0.902 (−25.9%) | 🟩 **2.078 (+70.8%)** | 1.400 (+15.1%) |
| 128 | 1.069 (0%) | 0.979 (−8.4%) | 🟩 **2.153 (+101.4%)** | 1.358 (+27.0%) |
| 512 | 1.077 (0%) | 0.928 (−13.8%) | 🟩 **2.062 (+91.4%)** | 1.196 (+11.0%) |
| 1024 | 0.913 (0%) | 0.919 (+0.6%) | 🟩 **2.072 (+126.8%)** | 1.014 (+11.0%) |
| 2048 | 0.897 (0%) | 0.940 (+4.9%) | 🟩 **2.080 (+131.9%)** | 0.808 (−9.9%) |

The [Mac expert-pack implementation](macos-expert-pack.md) selects pack reads only for single-token routed `MUL_MAT_ID` calls. Batched prefill still uses GGUF mmap, explaining the absence of a pack prefill benefit. The small original/pack prefill differences alternate in sign and are within run variation; they do not establish a pack penalty. At 1024 and 2048 input tokens, pack decode was 2.25× and 2.21× the original ssd-llama rate, respectively.

The original Q8 binary SHA256 was `5400a4899295bc8b2e5821bed5875e5d2fea5438f354f9be2a78f0224167ad27`, the pack binary `19ed367eff092710ed7b85d917488d0dc31fc1904891492dbde3aa5569bc1322`, and the fork binary `0c14fcd9b67eff22926aeb98b6bd217d7e1cafef9189275a1078c66b5caf7797`. The full server logs and response files remain in the MacBook Pro's `~/Desktop/ssd-llama/prefill-rerun-20261015/` and `~/Desktop/ssd-llama/long-prompt-20261015/` directories; the checked-in summaries preserve the exact rates and timings used here.
