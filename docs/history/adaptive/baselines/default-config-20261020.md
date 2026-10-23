# Baselines in their default configuration, three machines (2026-10-19/22)

llama.cpp and colibri serving DeepSeek-V4.1-Flash-1017 and Muse Spark Open on the three machines the adaptive
framework is meant for. Each baseline runs with its own defaults. The only changes are the ones it cannot run
without; they are listed with their evidence under [Changes needed to run](#changes-needed-to-run). Nothing is tuned.
Each prompt length is run once, and a run gets 60 minutes. This round uses new prompts and the 1017 weights, so its
numbers are not comparable with the 10-13 to 10-16 round in [README.md](README.md).

## Summary

Cells are tok/s: prefill / decode over tokens 1-128 / decode over tokens 129-256. "stopped" means the run reached
the 60-minute cap; its decode numbers then cover the tokens it produced.

| model, system | prompt | DGX Spark | 230F + RTX 5090 (125) | 100U + RTX 5060 Ti (233) |
|---|---|---|---|---|
| DSV-1017 Q8_0, llama.cpp PR #28696 | 512 | 2.39 / 1.406 / 1.453 | 6.15 / 0.825 / 0.790 | 5.51 / 0.246 / 0.246 |
| | 4096 | 2.73 / 1.320 / 1.608 | 8.06 / 0.873 / 0.867 | 6.95 / 0.241 / 0.242 |
| Muse UD-Q4_K_XL, llama.cpp upstream | 512 | 1.22 / 0.122 / 0.122 | 2.61 / 0.151 / 0.156 | 2.29 / 0.066 / -, stopped at 216 tokens |
| | 4096 | 1.74 / 0.120 / -, stopped at 141 tokens | 4.00 / 0.152 / 0.155 | 3.41 / 0.059 / -, stopped at 140 tokens |
| DSV-1017 official, colibri | 512 | 1.28 / 0.612 / 0.600 | 11.66 / 2.110 / 2.370 (Vulkan build) | 5.40 / 1.051 / 1.004 (Vulkan build) |
| | 4096 | still in prefill at 60 min | 39.21 / 1.645 / 1.878 (Vulkan build) | 7.02 / 0.897 / 0.900 (Vulkan build) |

The 125 (RTX 5090) box replaces 176 for this round; 176's 10-20 numbers are listed under [All runs](#all-runs). 125's
and 233's numbers are from 10-21, with both machines aligned with the 166 test machine (see [Machines](#machines)).
233's 10-20 llama.cpp runs, made before its alignment, are also listed under [All runs](#all-runs).
On 10-22 125 was measured again under the target machine's memory limit; see
[125 under the target-machine envelope](#125-under-the-target-machine-envelope-2026-10-22).

- **DSV-1017 on llama.cpp** finishes both prompts on 125 and 233 well inside the hour. The two machines prefill at
  similar speeds (125 6.2-8.1, 233 5.5-6.9 tok/s). 125 decodes 3.2-3.6 times faster (0.79-0.87 vs 0.24 tok/s); it has
  twice the VRAM and four times the RAM of 233. Both read the model through mmap page faults at 1.3-2.1 GB/s, from
  drives that read 6-12 GiB/s. The Spark also finishes both prompts within the hour, with the same two flags as Muse
  (see the next point). It prefills at 34-39% of 125's speed (2.4-2.7 tok/s) and decodes 1.5-1.9 times as fast
  (1.3-1.6 tok/s).
- **Muse on llama.cpp** decodes at 0.06-0.16 tok/s. Only 125 finishes the 4096 prompt plus 256 tokens in the hour
  (in about 46 minutes); the Spark and 233 reach the 60-minute cap. On the Spark it runs only with two flags: one maps the model
  instead of reading it into memory, the other gives `--fit` a larger memory margin.
- **colibri** on the two local machines runs its Vulkan build, the one colibri's own `coli setup` chooses with a
  discrete GPU. It moves the dense weights to the GPU and keeps an expert tier there. On 125 it runs exactly as its
  gateway configures it (cap 19, `CTX` unset). It is the fastest baseline on both machines: decode 1.6-2.4 tok/s on
  125, 0.9-1.05 on 233. On 233 the CPU build cannot load at all (7.3 GiB RAM), and the gateway refuses even the
  Vulkan build, so it runs with one expert slot and `CTX=8192`. On the Spark colibri runs its CPU build, which
  `coli setup` chooses on an integrated GPU, as its gateway configures it. It finishes the 512 prompt; the 4096
  prompt takes 83 minutes to prefill.

## Machines

| | DGX Spark (spark-b9a5) | 125 (RTX 5090, 31 GiB RAM) | 233 |
|---|---|---|---|
| CPU | GB10, 20 Arm cores (10 X925 + 10 A725) | Core Ultra 5 230F, 10 cores, no SMT | Core 3 100U, 2 P (HT) + 4 E |
| memory | 128 GB LPDDR5x unified (MemTotal 121.7 GiB), shared by CPU and GPU | 30.9 GiB | 7.3 GiB |
| GPU | GB10 integrated, no memory of its own | RTX 5090 32 GB, PCIe Gen5 x16, 575 W | RTX 5060 Ti 16 GB, PCIe Gen4 x4 |
| drive | Samsung MZALC4T0HBL1 4 TB, Gen5 x4; 9.6-11.0 GiB/s before each run | ZHITAI TiPro9000 2 TB, Gen5 x4; 11.3-11.5 GiB/s before each run | ZHITAI TiPro9000 2 TB, Gen4 x4; 6.2-6.6 GiB/s before each run |
| swap | none | 8 GiB file (the Ubuntu installer's size for 31 GB of RAM), swappiness 60 | 4 GiB file, swappiness 60 |
| software | Ubuntu 24.04.4, kernel 6.17.0-1029-nvidia, driver 580.173.02, CUDA 13.0 | Ubuntu 24.04.4, kernel 7.0.0-28, driver 595.71.05, CUDA 13.2 | kernel 7.0.0-31, driver 595.99.02, CUDA 13.0 |

The drive speed is a 4 x 1 GiB O_DIRECT read taken before each run. On the local machines the machine ran nothing else:
Other services were stopped, and additional swap files and network buffer settings were back at the system
defaults. ssd-llama's host settings stayed (performance power profile, GPU persistence and clocks, `intel_iommu=off`
on 176). 176's Gen5 drive slows down when hot, so each run there started with the drive below 55 C.

125's results are from 10-21 17:28-19:35 CST, in the 5090's native configuration: no H2D limiter, no memory or
swap cap. Its system configuration on 10-21 was:
- `intel_iommu=off`;
- governor `performance`;
- NVMe `pm_qos_latency_tolerance_us` 0;
- `cpu_dma_latency` 0, held by a service;
- GPU persistence on, with the clock locked at 3090 MHz.

PCIe ASPM is off in its BIOS, so the kernel's `pcie_aspm` policy (`default`) does not apply. The test user's slice
(`user-1000.slice`) carried `MemoryMax=15 GiB` and `MemorySwapMax=0` from 08-04, a leftover 16 GB-box envelope. It
was set to `infinity` before these runs; the old drop-ins are saved in `~/bl125/runs-8b/user-1000.slice.envelope-before.txt`.
The five runs made under it (16:06-17:25) are not results; colibri's 4096 run among them was killed by that cgroup
(`CONSTRAINT_MEMCG`) with about 14 GiB still free. Each 125 run started once the hottest drive sensor was below 62 C.

233's results are from 10-21 02:50-07:19 CST. Its system configuration was set on 10-20 at about 21:00:
- kernel command line `intel_iommu=off pcie_aspm.policy=performance`, which also raises the NVMe `max_hw_sectors_kb`
  to 512;
- governor `performance` (it was `powersave` with EPP `performance`);
- NVMe `pm_qos_latency_tolerance_us` 0;
- `cpu_dma_latency` 0, held by a service;
- GPU persistence on, with the clock locked at 3090 MHz.

A control run of llama.cpp's DSV-1017 512 arm, unchanged otherwise, came out 17% faster in prefill (5.51 vs 4.72 tok/s)
and 9% faster in decode (0.246 vs 0.226 tok/s) than on 10-20, with a byte-identical reply. All four llama.cpp arms on
233 were therefore rerun; the 10-20 runs are listed under [All runs](#all-runs). Each 233 run on 10-21 started once the
hottest of the drive's three temperature sensors was below 62 C. One of them idles about 10 C above the composite
reading, and SMART shows the drive has never throttled.

On the Spark the models are on the drive's own file system (`/media/golden-root/var/tmp`). The login environment's root
is an overlay that reads at about 1 GB/s and keeps a second page-cache copy, so it was not used. During the Muse
and colibri runs the only other process was the stalled download of the DSV-1017 GGUF, writing under 1 MB/s. The
download had finished before the DSV-1017 llama.cpp runs (10-21 00:31-01:48 CST), and nothing else ran during them.

## Software and models

| baseline | source | build |
|---|---|---|
| llama.cpp for Muse | ggml-org/llama.cpp master `4866db350d619187ee71b85e56f0b772db4dbf1d` (has `muse-spark`) | `cmake -B build-cuda -DGGML_CUDA=ON -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release`, then `cmake --build build-cuda --target llama-server`; native CUDA arch |
| llama.cpp for DSV-1017 | vcruz305/llama.cpp `2076412598e99127ea2554467b2b7530856895d3` (PR #28696, by the GGUF's author) | same |
| colibri | JustVugg/colibri main `e0db083cdc9e563e0766cc6fc661ef23c32298e0` | Spark and 176: `make -C c -j deepseek_v41` (the default CPU build). 233 and 125: `make -C c deepseek_v41 VK=1` (the Vulkan build, the one `coli setup` makes with a discrete GPU; the 35 committed shaders kept, 5 compiled by the system's glslc, tree clean) |

No source tree was changed. OpenSSL and curl are off only because the Spark lacks their development packages; the
local machines used the same line. 233 built with `-j 2`, 176 with `-j 4` and 125 with `-j 6`, because more parallel nvcc jobs run 233 out of memory; this only changes
the build time. 125 builds with CUDA 13.2 from `/usr/local/cuda`; the distribution's `/usr/bin/nvcc` is 12.0, which has
no sm_120.

Models, read in place:
- Muse: `unsloth/Muse-Spark-Open-816B-A42B-GGUF`, UD-Q4_K_XL, 12 shards, 505 GB (experts 483 GB: gate/up Q4_K, down
  Q5_K). The Spark's copy came from ModelScope's mirror of the repository; the sha256 of all 12 shards equals the LFS
  hashes of Hugging Face revision `c00670568b6ce5e56a247f67091247984df2a01e`.
- DSV-1017 Q8_0: `mrfakename/DeepSeek-V4.1-Flash-1017-GGUF`, Q8_0, 10 shards, 508 GB. The routed experts are MXFP4,
  bit-identical to the official FP4 weights in sampled layers. The Spark's copy is revision
  `1df54e7d1b1e442051ad393ce871870a929f243e`, downloaded through hf-mirror.com; the sha256 of all 10 shards equals
  that revision's LFS hashes. Shards 8 and 9 (one 104 GB engram table each) were fetched from Hugging Face's
  `eu-west-3.aws.cdn.hf.co` with the signed URL the mirror redirects to. The redirect itself names
  `us.aws.cdn.hf.co`, which serves from Singapore and gave the Spark about 0.06 MB/s.
- DSV-1017 official: `deepseek-ai/DeepSeek-V4.1-Flash-1017`, 48 safetensors shards, 510 GB, for colibri. The Spark's
  copy came from ModelScope; the sha256 of all 48 shards equals the LFS hashes of Hugging Face revision
  `1bae0e1088993b9b82de08f59de5be5a34e42170`. colibri's one-time preparation
  (`python c/tools/prepare_dsv41.py --model <dir>`) writes `dsv41_engram.json` (876,024 bytes, sha256 `fb92589b...`)
  next to `config.json`. colibri's planner also caches its model analysis there as `.coli_analysis.json` (190,465
  bytes on 176 and 233, 190,480 on the Spark, 190,454 on 125). These are the only files written into a model directory.
- 125's copies of all three sets came from a USB drive. Their sha256 (git sha1 for small files) equals the Hugging Face
  hashes above for all 105 files. Its `dsv41_engram.json` was copied from 233 (sha256 `fb92589b...`, as everywhere).

## Protocol

- Prompts: the unified `prompt-512.txt` (sha256 `17ac26b4a3bc13d6c51dd10b62eb54931651d48e41ca4071c587a931a662e283`)
  and `prompt-4096.txt` (sha256 `7eac0ca46f379476846f8221e2ab9617d3023c05f3655769eed523a101754dec`), cut from one
  invented English design document that no model has seen. One user message. Rendered, they are 523 / 4104 tokens
  for DSV-1017 and 669 / 4221 tokens for Muse (Muse's template adds a system message).
- Thinking off, greedy (temperature 0, top_k 1), 256 new tokens, EOS ignored (colibri has no such option, but
  produced 256 tokens), no prompt cache.
  - DSV-1017 on llama.cpp: `/apply-template` with `chat_template_kwargs: {"enable_thinking": false}`, then
    `/completion`. The rendered prompt ends with `<｜Assistant｜></think>` and has exactly one BOS (checked with
    `/tokenize`). The GGUF carries the DeepSeek-V4 template, which renders a plain chat turn the same way as V4.1's.
  - Muse on llama.cpp: the template has no switch for thinking. The client renders with the server's template
    (`current_date` fixed to 2026-10-19), appends ` to=user<|message|>` (how the official template renders an
    answer without reasoning) and generates with `/completion`.
  - colibri: its native protocol, with its own non-thinking render
    `<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜></think>`. The engine is started the way colibri's
    gateway starts it: `SERVE=1 SERVE_BATCH=1 KV_SLOTS=1 NGEN=256`, `OMP_NUM_THREADS` = the launcher's physical-core count (16 on
    176, 6 on 233, 10 on 125 and on the Spark), `CTX` unset,
    and the expert-slot cap the gateway computes on the idle machine
    (`build_plan(model, ram_gb=0, context=4096, gpu_indices=[])`). On 233 and 125 the Vulkan build also gets
    `COLI_VULKAN=1`, which `coli setup`'s run environment sets, and `COLI_USAGE` pointing at a new file for each run.
    With the expert tier on, the engine keeps an expert history there (default `<model>/.coli_usage`) and the next
    run would warm-start from it. A new file keeps every run cold and writes nothing into the model directory.
- Cold start: `posix_fadvise(DONTNEED)` on every model file, then the drive read, then the server starts.
- Metrics: prefill tok/s = the server's `prompt_per_second` (llama.cpp) or prompt tokens / time to first token
  (colibri). Decode 1-128 = 127 token intervals from token 1 to token 128; decode 129-256 = 128 intervals from token
  128 to token 256.
- Time limit: 60 minutes from sending the request (colibri: from SUBMIT), and 30 minutes for loading the model. At
  the limit the client is stopped and keeps what it reached: time to first token, the decode rate over the tokens it
  has, and the token count; for llama.cpp still in prefill also the server's last prefill progress line. colibri
  prints nothing during prefill, so a colibri run cut there is recorded as "still in prefill at 60 min".
- A watchdog stops the server if the machine is about to run out of memory. On 125, 176 and 233: MemAvailable below
  1 GiB, swap use growing by more than 4 GiB, or more than 1 GiB swapped in within 10 seconds. On the Spark, which has
  no swap: MemAvailable below 3 GiB, checked every 0.25 s, then the server is killed (SIGKILL). The floor was 1 GiB for
  the first default Muse run. It is set higher because the GPU's allocations on the unified memory are in no cgroup,
  and the machine's operator warns that running out of memory can reboot it.
  - Exception: colibri's Vulkan runs on 233 used a MemAvailable floor of 128 MiB and a swap-growth limit of 3 GiB;
    the swap-in limit stayed at 1 GiB per 10 s. These are harness settings, not colibri's.
  - With the 1 GiB floor the 512 run was stopped during prefill at 1013 MiB (`20261021-024956`), before the kernel had
    moved anything to swap. With a 256 MiB floor it was stopped at 236 MiB (`20261021-025321`), still with almost
    nothing swapped out.
  - The kernel moves cold anonymous pages (the embedding, the vision tower, the host copy of the head) to swap only
    near its low watermark, about 75 MB free.
  - With the 128 MiB floor the runs bottomed out at 0.38 GiB (512) and 0.21 GiB (4096). They used 0.28 and 2.18 GiB
    of swap and swapped 12 and 998 MiB back in over the whole run, without any thrashing.

## Changes needed to run

| machine | arm | change | why |
|---|---|---|---|
| all | llama.cpp, 4096 prompt | `-c 8192` | Without `-c`, `--fit` sets 4 slots of 4096 tokens (log: `n_slots = 4, n_ctx_slot = 4096`), and 4104 + 256 (DSV) or 4221 + 256 (Muse) tokens do not fit in one slot. The 512 prompt runs without `-c`. |
| Spark | Muse, DSV-1017 | `--load-mode mmap` | GB10's integrated GPU reports `mmap_support = false` (`ggml-cuda.cu`), and with `--load-mode auto` the loader then turns mmap off (`llama-model.cpp`), so the 505 GB (Muse) or 508 GB (DSV-1017) model would be read into 121.7 GiB of memory. The PR #28696 branch has the same code (`ggml-cuda.cu:5043`, `llama-model.cpp:1446`). This follows from the code and the sizes; it was not run. On 176 and 233 auto already means mmap. |
| Spark | Muse | `--fit-target 7168` | With `--load-mode mmap` and the default `--fit` margin of 1 GiB, `--fit` treats the unified memory as free VRAM and fills it. MemAvailable fell from 116 GB to 0.8 GiB 798 s into the load, and the watchdog stopped the run (that run also had `-c 8192`). 7168 MiB is the smallest margin that keeps MemAvailable above 3 GiB. It was derived from a 512 run with 16384, whose minimum MemAvailable was 13.08 GiB: 16 - (13.08 - 3) = 5.9 GiB, plus 1 GiB of headroom. The runs with 7168 bottomed out at 5.71 GiB (512) and 4.05 GiB (4096). The 16384 run decoded at 0.139 / 0.140 tok/s on the 512 prompt, faster than 7168, because a larger margin leaves more page cache for the experts on the CPU. |
| Spark | DSV-1017 | `--fit-target 7168` (512 run made with 16384, which gives the same fitted arguments) | With `--load-mode mmap` and the default margin of 1 GiB, `--fit` puts layer 14 on the GPU, and with it that layer's 104 GB engram table. That run (`20261021-004552-dsv-mmap-p512`) was still loading after 1348 s when MemAvailable fell to 2.7 GiB and the watchdog stopped it. The placement does not change smoothly with the margin, so the Muse derivation does not apply. `llama-fit-params` (same build, same machine) prints the arguments `--fit` would choose. Without `-c`, every margin from 4096 to 16384 MiB gives byte-identical arguments: `-c 4096 -ngl 26`, plus `-ot` rules for the routed experts of layers 16-39. The model has 40 layers, so layers 15-39 and the output layer go to the GPU, with the experts of layers 16-39 kept on the CPU. Layers 0-14, including both engram tables (layers 1 and 14), stay on the CPU. `-ngl 27` or more takes layer 14 and its table to the GPU (projected 114,939 MiB instead of 14,973), and 3584 MiB or less gives `-ngl 29`. With `-c 8192`, 7168 and above give the 16384 placement, and 4096 gives `-ngl 28`, which already includes layer 14. 7168 is therefore the smallest margin with that placement in both tiers; it is also the value Muse uses. The 512 run was made with 16384 (`20261021-003113-dsv-fit16g-p512`), whose fitted arguments are identical to 7168's. The 4096 run used 7168 with `-c 8192` (`20261021-011227-dsv-fit7g-c8k-p4096`). Both runs used 15.3 GiB of GPU memory, and MemAvailable stayed above 99.7 GiB. |
| 176, 233 (not 125) | colibri | expert-slot cap 1 instead of the gateway's 0 | The gateway refuses to start: `cache_slots_per_layer = 0`, "RAM budget cannot hold one expert slot per sparse layer". On 176 the budget is 13.0 GB for 11.5 GB of dense weights plus 5.8 GB of runtime buffers; on 233 it is 6.0 GB. With `COLI_VULKAN=1` the planner on 233 credits the GPU with 6.88 GB of the dense weights (all 40 layers in the device chain), but 4.65 GB of dense weights plus 5.78 GB of runtime buffers still exceed the 5.91 GB budget, so the cap stays 0 (`~/bl233/runs-8b/gwcap-vk-*.txt`). 1 is the smallest cap the engine accepts. On 125 the gateway's own plan with `COLI_VULKAN=1` is 19 slots per layer (budget 26.5 GB), so no cap is set by hand (`~/bl125/runs-8b/gwcap-vk-noenvelope.txt`). |
| 176, 233 (not 125) | colibri | `CTX=8192` | `config.json` gives `max_position_embeddings` 1,048,576, and with `CTX` unset the engine sizes its buffers for that many positions; the KV and rope buffers come to about 6.75 GiB (`deepseek_v41.c:206-226`). On 176 MemAvailable fell from 13.8 to 0.83 GiB in 7-8 seconds, before the engine was ready, on both prompts. With `CTX=8192` it loads in 5.8 s and holds 12.4 GB. On 233 those buffers alone exceed the free memory. On 125 the engine runs with `CTX` unset as the gateway starts it: RSS 11.3 GiB after loading, and MemAvailable stayed above 1.18 GiB. |
| 233, 125 | colibri | the Vulkan build (`VK=1`) with `COLI_VULKAN=1` | The CPU build reads all 11.5 GB of dense weights into anonymous memory (`xmalloc` and `read`). With cap 1 it was stopped 3 seconds after start, when MemAvailable fell below 1 GiB, both with `CTX` unset and with `CTX=8192` (runs `20261020-023208`, `-023553`, `-041325`). With `VK=1` and `COLI_VULKAN=1` on a discrete GPU, the engine opens the device before it reads the weights. It then reads the dense matrices one layer at a time, uploads them, and gives their pages back (`deepseek_v41.c:1972`, `:2065`, `:391-402`). Log: "349 dense matrices on the device only, 6.40 GiB of host RAM given back ... RSS now 4.60 GiB". This is the backend colibri's own installer selects here. `coli setup --backend auto` builds `VK=1` and starts with `COLI_VULKAN=1` when there is a discrete Vulkan GPU, glslc and the Vulkan headers, because the V4.1 engine has no CUDA path (`setup_flow.py:433-441`, `:580-585`, `:676-684`, `:1090-1097`). The gateway (`coli serve`) never sets it by itself. On 125 the CPU build would also start (its gateway plan is 11 slots per layer); the Vulkan build is used there because it is what `coli setup` picks with a discrete GPU, as on 233. The Spark's GB10 is an integrated GPU, where `coli setup` keeps V4.1 on the CPU (`setup_flow.py:573-579`). |

Spark's colibri needed no change: the gateway's cap there is 115. Its 4096 run is an earlier one with cap 113,
`CTX=8192` and `V41_STATS=1` (statistics printing only) and no time limit. On the 512 prompt that configuration and
the strict default agree within 2% (prefill 1.268 vs 1.280, decode 0.601 vs 0.612 tok/s). The run reached its first
token after 4966 s, so it is reported as still in prefill at 60 minutes; for reference it prefilled at 0.826 tok/s and
decoded at 0.349 / 0.356 tok/s.

## All runs

| machine | arm | prompt | load s | time to first token s | tokens | result |
|---|---|---|---|---|---|---|
| Spark | Muse | 512 | 802 | 548 | 256 | complete |
| Spark | Muse | 4096 | 820 | 2424 | 141 | stopped at 60 min |
| Spark | DSV | 512 | 477 | 219 | 256 | complete (`--load-mode mmap --fit-target 16384`) |
| Spark | DSV | 4096 | 465 | 1503 | 256 | complete (`--load-mode mmap --fit-target 7168 -c 8192`) |
| Spark | colibri | 512 | 13.5 | 409 | 256 | complete |
| Spark | colibri | 4096 | 11.5 | 4966 (no limit) | 256 | reported as still in prefill at 60 min |
| 125 | Muse | 512 | 230 | 256 | 256 | complete |
| 125 | Muse | 4096 | 235 | 1055 | 256 | complete |
| 125 | DSV | 512 | 131 | 85 | 256 | complete |
| 125 | DSV | 4096 | 126 | 509 | 256 | complete |
| 125 | colibri (Vulkan) | 512 | 12.6 | 45 | 256 | complete |
| 125 | colibri (Vulkan) | 4096 | 12.2 | 105 | 256 | complete |
| 233 | Muse | 512 | 239 | 292 | 216 | stopped at 60 min |
| 233 | Muse | 4096 | 240 | 1238 | 140 | stopped at 60 min |
| 233 | DSV | 512 | 138 | 95 | 256 | complete |
| 233 | DSV | 4096 | 131 | 591 | 256 | complete |
| 233 | colibri (Vulkan) | 512 | 8.5 | 97 | 256 | complete |
| 233 | colibri (Vulkan) | 4096 | 8.5 | 584 | 256 | complete |

176 (255H, 15.2 GiB, RTX 5090 on PCIe Gen4 x4), replaced by 125 for this round; its 10-20 runs (prefill / decode 1-128 /
decode 129-256), with the changes it needed (colibri cap 1 and `CTX=8192`, CPU build):

| arm | prompt | load s | TTFT s | result |
|---|---|---|---|---|
| DSV | 512 | 160 | 119 | 4.40 / 0.577 / 0.536 |
| DSV | 4096 | 156 | 740 | 5.55 / 0.552 / 0.551 |
| Muse | 512 | 286 | 345 | 1.94 / 0.101 / 0.099 |
| Muse | 4096 | 288 | 1539 | 2.74 / 0.093 / 0.092, stopped at 60 min with 192 tokens |
| colibri | 512 | 5.8 | 621 | 0.84 / 0.550 / 0.411 |
| colibri | 4096 | 5.8 | - | stopped by the memory guard at 44.9 min, still in prefill |

233's 10-20 llama.cpp runs, made before the 166 alignment, are superseded. Prefill / decode 1-128 / decode 129-256:

| arm | prompt | 10-20 run | result |
|---|---|---|---|
| DSV | 512 | `20261020-020853` | 4.72 / 0.226 / 0.225 |
| DSV | 4096 | `20261020-034037` | 6.12 / 0.224 / 0.222 |
| Muse | 512 | `20261020-010428` | 2.03 / 0.063 / 0.063, 207 tokens |
| Muse | 4096 | `20261020-023558` | 2.97 / 0.055 / -, 120 tokens |

The 10-21 runs end in `-aligned166`.

176's colibri 4096 run was not repeated. Its memory grew during prefill from 2.0 GiB free after loading to below
1 GiB, without touching swap. At its 512 prefill rate (0.84 tok/s) the 4104 tokens would take about 81 minutes, so it
would most likely have had no first token at 60 minutes either; this is an estimate.

One run was repeated: 176's first colibri 512 run was stopped 6 seconds after start by a guard on swap growth (then
3 GiB) while 9.4 GiB was still free. The same configuration loaded on the 4096 prompt without any swap, so the run
was counted as invalid and rerun with the 4 GiB guard 233 uses. Runs from before the procedure was aligned on 10-20
01:00, and runs with the earlier system settings, are kept in the run directories with their labels and are not results. Neither
is the Spark's DSV-1017 512 run with the default `--fit` margin (`20261021-004552-dsv-mmap-p512`, stopped by the
watchdog during loading): it is the evidence for the `--fit-target` change. The same holds for 233's two colibri
Vulkan 512 attempts that the harness's MemAvailable floor stopped (`20261021-024956`, `20261021-025321`). They are the
evidence for the lower floor described under [Protocol](#protocol). 125's five runs under the 15 GiB slice envelope
(16:06-17:25, see [Machines](#machines)) are not results either.

## Notes

- llama.cpp's default thread count is low on the Intel machines: 3 on 125 and 176, 2 on 233. Its heuristic counts the
  physical P-cores, and on the 255H it also skips every second core as an assumed SMT sibling, although the 255H has
  no SMT. On the Spark it uses all 20 cores. The default was kept.
- colibri's launcher sets `OMP_NUM_THREADS` to its physical-core count. That count is 10 on the Spark: its kernel
  numbers the 20 cores with 10 core ids (two CPUs per id, as SMT siblings would be), so colibri runs 10 threads on
  20 cores. The default was kept. The V4.1 engine reads none of the other variables the launcher adds there
  (`COLI_POLICY`, `RAM_GB`, `COLI_PLAN_CAP`).
- On the Spark, `--fit` puts part of Muse's experts in GPU memory, which is the same memory as the page cache that
  holds the experts left on the CPU. The margin therefore trades one for the other, which is why the 16 GiB margin
  decoded faster than 7 GiB.
- For DSV-1017 on the Spark, `--fit` keeps only 15.3 GiB on the GPU: layer 15 with its experts, the non-expert
  weights of layers 16-39, and the output layer. The rest is read through mmap and stays in the page cache;
  MemAvailable was about 100 GiB in both runs. Loading read the
  whole model from the drive (about 525 GB in 465-477 s). The drive then read 0.9-1.0 GB/s during prefill and
  0.7-0.8 GB/s during decode.
- colibri's V4.1 engine drafts tokens with the checkpoint's three DSpark stages by default. In the Spark runs that
  print statistics, 8 of 40 drafts (512) and 6 of 40 (4096) were accepted.
- colibri's prefill on the Spark is bound by CPU work rather than by the drive. The statistics of the earlier
  `V41_STATS=1` runs show 355 GB of experts read during the 512 prefill, with 47 s on the drive out of 411 s. For the
  4096 prefill it was 2.25 TB, with 294 s on the drive out of 4956 s.
- llama.cpp on 176 and 233 reads the experts that `--fit` leaves on the CPU through mmap page faults: 1.5-1.6 GB/s on
  176, during decode 1.31-1.43 GB/s on 125 and 1.75-2.13 GB/s on 233 (1.61 GB/s in 233's 10-20 DSV 512 run). On 125
  the kernel also moved 0.7-1.5 GiB of llama-server's memory to swap while MemAvailable stayed at 25-28 GiB
  (swappiness 60 with the model streaming through the page cache). On 233 the kernel also swaps llama-server's own memory out and back in while the model
  streams (25-570 MiB per decoded token); that is 0.4-1.8% of the 7-34 GB read per token.
- colibri with one expert slot per layer reads every routed expert from disk: 3.66 TB in 176's 512 run (3.15 GB/s).
  176's drive reached 77-81 C in the colibri runs, and the 512 run's decode fell from 0.55 to 0.41 tok/s in its second
  half.
- On 176, colibri's 512 reply (official weights) and llama.cpp's DSV-1017 Q8_0 reply are identical for the first 881
  characters.
- colibri's Vulkan build on 233:
  - **GPU:** it holds 9.2 GiB of matrices on the 16 GB card (the dense chain, all 40 layers) plus an expert tier of
    177 experts (5.41 GiB). It computes 50.5% (512) and 71.4% (4096) of the routed experts there; the CPU computes the
    rest, from a one-slot-per-layer RAM cache and the drive.
  - **Host memory:** RSS is 4.6 GiB after loading. Kept on the host are the embedding, the head, the routers, norms
    and mixers, DSpark's stages and the vision tower.
  - **Load and DSpark:** it loads in 8.5 s, and DSpark drafting stays on.
  - **Speed:** it decodes at 1.0-1.05 (512) and 0.9 (4096) tok/s. That is 4 times llama.cpp on the same machine, and
    above colibri's CPU build on 176 (0.41-0.55).
- colibri's Vulkan build on 125 (gateway cap 19, `CTX` unset):
  - **GPU:** the same 9.2 GiB dense chain plus an expert tier of 883 experts (21.2 GiB). It computes 59.3% (512) and
    90.5% (4096) of the routed experts on the GPU; the rest come from 19 RAM slots per layer and the drive.
  - **Host memory:** RSS is 11.3 GiB after loading (KV buffers for 1,048,576 positions) and 25.5-25.7 GiB at exit.
  - **Speed:** it prefills the 4096 prompt at 39 tok/s (prompt chunks up to 6,656 rows on the GPU) and decodes at
    1.6-2.4 tok/s, 1.9-3.0 times llama.cpp's DSV-1017 decode on the same machine.

## 125 under the target-machine envelope (2026-10-22)

The target machine is a 16 GB box with an RTX 5090 on PCIe Gen4 x4. On 10-22 125 was measured again under that
machine's memory limit, as the 166 test machine and 125 itself had emulated it before (13:55-16:09 CST for the baselines,
16:11-16:29 for ssd-llama).

**Envelope.** Each server runs in its own systemd user scope with `MemoryMax=13976514560` (13.98 GB, the target
deployment's limit) and `MemorySwapMax=0`, so the model's page cache counts against the limit as well. The user slice
stays unlimited. llama.cpp's server is the scope's process. For colibri the scope holds only the engine (a per-run exec
wrapper given to the client as the engine binary), not the Python client. ssd-llama's launcher makes its own scope from
`MEMORY_MAX`. All 10 cores are used (the target has 16 threads), and GPU and drive are as they are.

**Host-to-device link: not limited.** All three systems use 125's PCIe Gen5 x16, so this is optimistic against the
target's Gen4 x4 for the systems that move experts over the link. The cap was dropped (decision, 10-22 14:12)
for three reasons:
- **The shim cannot reach colibri.** The H2D governor (`libh2dgov`, 6.74 GB/s) paces the CUDA runtime's copies and
  cannot pace colibri's Vulkan engine. That engine does use the link: its prefill streams cold experts to its 21.2 GiB
  device tier at 5-11 GB/s, peaking at 12.8 GB/s (nvidia-smi `dmon`).
- **It would handicap ssd-llama alone.** A cap that only CUDA programs see would slow ssd-llama and leave colibri at
  full speed.
- **It does not bind llama.cpp.** The first two llama.cpp arms (DSV-1017 512 and 4096) ran with the governor
  preloaded. Their copies, from mmap page faults under the memory limit, averaged 2.27 and 2.22 GB/s, and no 2-second
  window went above 2.59 GB/s. The cap was never reached, so these two arms count as memory-only runs. The later arms
  preload nothing.

**Changes needed under the envelope.** As before, only what a system cannot run without:

| arm | change | why |
|---|---|---|
| colibri, both prompts | `CTX=8192` | With `CTX` unset (the gateway's start) the engine was killed by the scope's OOM killer while loading, at 13.8 GB of anonymous memory (`CONSTRAINT_MEMCG`; runs `20261022-143112`, `-143825`). This is the change 233 needed. |
| colibri | none for the cap | The gateway's own plan inside the scope is 2 slots per layer with `COLI_VULKAN=1` (budget 12.2 GB); the CPU build's plan is 0 and the gateway refuses it (`runs-env/aux/gwcap-vk*-envelope.txt`). |
| llama.cpp, Muse | stopped once decode was stable (decision, 10-22 15:2x) | At about 0.08 tok/s the 256 tokens would not fit in the hour. The 512 run was stopped at 148 tokens: its decode over tokens 100-140, in 8-token blocks, was 0.085-0.094 tok/s. The 4096 run was stopped at 96 tokens, by a rule fixed beforehand: at least 48 tokens, and the last three 16-token blocks within 10 % of their mean (0.084 / 0.080 / 0.082). |

ssd-llama is not a baseline. It is listed for comparison, with the settings found best on 125 in its native
configuration, not derived again under the envelope (decision, 10-22 16:0x), and only `MEMORY_MAX` added:
- DSV-1017: engine `6678569` with its launcher's profile and model configuration (no settings file).
- Muse: engine `135f775` with the framework's 125 settings (`CONTEXT=6400`, `UBATCH=4864`,
  `GGML_KIMI_CPU_PREFILL_MAX_TOKENS=331`, `GGML_KIMI_CPU_SINK_PREFETCH=2`, `CTX_CHECKPOINTS=1`).
- The RAM expert cache sizes itself from the scope's limit.

**Results.** Each cell is prefill / decode 1-128 / decode 129-256 in tok/s, with the time to first token in
parentheses. The native column is the 10-21 configuration from the [Summary](#summary); ssd-llama's native numbers are
the same engines and settings (DSV-1017: typical values of the about 20 native arms per prompt run on 10-22).

| model, system | prompt | 125 envelope (13.98 GB, no swap) | 125 native |
|---|---|---|---|
| DSV-1017 Q8_0, llama.cpp PR #28696 | 512 | 5.80 / 0.538 / 0.468 (90 s) | 6.15 / 0.825 / 0.790 (85 s) |
| | 4096 | 7.37 / 0.516 / 0.526 (557 s) | 8.06 / 0.873 / 0.867 (509 s) |
| Muse UD-Q4_K_XL, llama.cpp upstream | 512 | 2.42 / 0.086 / -, stopped at 148 tokens (276 s) | 2.61 / 0.151 / 0.156 (256 s) |
| | 4096 | 3.77 / 0.080 over tokens 1-96, stopped at 96 tokens (1120 s) | 4.00 / 0.152 / 0.155 (1055 s) |
| DSV-1017 official, colibri (Vulkan) | 512 | 10.91 / 1.750 / 1.764 (48 s), cap 2, `CTX=8192` | 11.66 / 2.110 / 2.370 (45 s), cap 19 |
| | 4096 | 36.81 / 1.242 / 1.342 (111 s), cap 2, `CTX=8192` | 39.21 / 1.645 / 1.878 (105 s), cap 19 |
| DSV-1017 Q8_0, ssd-llama `6678569` | 512 | 33.08 / 4.541 / 5.765 (15.8 s) | 33.2 / 5.78 / 6.71 (15.8 s) |
| | 4096 | 141.99 / 4.519 / 6.232 (28.9 s) | 164.5 / 5.77 / 7.27 (25.0 s) |
| Muse UD-Q4_K_XL, ssd-llama `135f775` | 512 | 20.80 / 1.201 / 1.280 (32.2 s) | 18.4 / 1.40 / 1.48 (36.3 s) |
| | 4096 | 112.33 / 1.258 / 1.286 (37.6 s) | 100.6-106.6 / 1.42-1.46 / 1.48-1.52 (39.6-42.0 s) |

**What the memory limit changes.**
- **Prefill** of llama.cpp and colibri is 6-9 % slower. llama.cpp streams the whole expert set from the drive for
  every batch, whether the page cache holds 12 or 25 GB of a 470-508 GB model.
- **llama.cpp decode** loses 35-41 % (DSV-1017) and 43-47 % (Muse). On Muse the drive read 1.37 GB/s in the native
  decode and 1.39 GB/s under the envelope, so each token needed about twice as many bytes from the drive: the native
  page cache of about 25 GB had served a large share of them.
- **colibri decode** loses 17-29 %: 2 expert slots per layer in RAM instead of 19, and `CTX=8192`.
- **ssd-llama DSV-1017 decode** (one envelope arm per prompt): tokens 1-128, at 4.52-4.54 tok/s, are below every
  native arm (4.91-5.96), about 22 % under their typical value. Tokens 129-256 (5.77 and 6.23) are inside the native
  arms' spread (5.06-7.39). Its RAM cache shrinks from 29.0 GB to 12.2 GB (about 650 slots), and the RAM hits of the
  512 run fall from 50 % to 41 %. The VRAM tier is unchanged (32.8k and 32.3k hits). Greedy answers diverge between
  runs, so a single arm's decode also carries a content part; a teacher-forced comparison would separate the two.
- **ssd-llama DSV-1017 4096 time to first token** grows by 16 % (28.9 s against 25.0 s). Experts read during prefill
  are admitted to the RAM cache and compete with the engram tables' page cache: the scope hit its limit 2,682 times in
  that run, 0 times in the 512 run. That is the likely cause, but one arm does not prove it: 4 of the about 20 native
  4096 arms also took 26.2-28.9 s, when the drive slowed.
- **ssd-llama Muse decode** loses 12-14 %. Its time to first token is not longer.
- **Against the best baseline under the envelope:**
  - DSV-1017: ssd-llama decodes 2.6-4.6 times as fast as colibri and 8-12 times as fast as llama.cpp, and prefills
    3-4 times colibri and 6-19 times llama.cpp.
  - Muse: 14-16 times llama.cpp's decode and 9-30 times its prefill.

| arm (125 envelope) | prompt | load s | time to first token s | tokens, result |
|---|---|---|---|---|
| llama.cpp DSV | 512 | 154 | 90 | 256, complete (governor preloaded, never at its cap) |
| llama.cpp DSV | 4096 | 149 | 557 | 256, complete (governor preloaded, never at its cap) |
| colibri (Vulkan), `CTX` unset | 512 / 4096 | - | - | killed by the scope's OOM killer while loading (evidence for `CTX=8192`) |
| colibri (Vulkan), `CTX=8192` | 512 | 8.5 | 48 | 256, complete |
| colibri (Vulkan), `CTX=8192` | 4096 | 8.6 | 111 | 256, complete |
| llama.cpp Muse | 512 | 262 | 276 | 148, stopped when decode was stable |
| llama.cpp Muse | 4096 | 263 | 1120 | 96, stopped when decode was stable |

## Raw data

Every run directory has the full command line, environment, binary hashes, the drive read, the server or engine log,
a 1 Hz monitor (disk, memory, swap, GPU, drive temperature on 176), the per-token times and the reply. The colibri Vulkan
runs also have a 1 Hz GPU memory log in `runs-8b/`.

- Spark: `~/bench/runs/` on spark-b9a5.
- 125: `~/bl125/runs/` (harness in `~/bl125/tools/`, a copy of 233's with paths changed).
- 176: `~/bl176/runs/`.
- 233: `~/bl233/runs/`.
- 125 envelope (10-22): `~/bl125/runs-env/` (harness `~/bl125/tools-env/`, chain
  `~/bl125/run-env-125.sh`, `decode-stop.py` for the Muse stop rule; each run also has `memcg.txt`, the
  scope's memory counters each second, and `pcie.txt`, nvidia-smi's PCIe throughput). ssd-llama:
  `~/ssdl2e/ab/runs/1022-161116-dsv-env-p512`, `-161234-dsv-env-p4096`, `-161444-muse-env-p512`,
  `-162042-muse-env-p4096`.
