# Benchmark methodology

[Back to 4K results](../README.md#performance-benchmarks)

The README comparisons use the **4K prompt tier**. This is the input package's label, not the context capacity or an exact token count: chat templates add tokens. All README decode rates use **tokens 129–256 of a 256-token output**. The tester confirmed this convention for RTX 5060 Ti, RTX 5090, and DGX Spark on **2026-10-25**. Mac runs use an **8192-token context**, one request, greedy sampling, no prompt-cache reuse, and EOS suppression. Device specifications are listed in [Test hardware](../README.md#test-hardware).

## Measurement definitions

| Metric | Definition |
| --- | --- |
| Prefill throughput | Actual prompt tokens / prompt-processing seconds |
| First 128 output tokens, new Mac runs | `127 / (t128 - t1)` |
| Last 128 output tokens | `128 / (T256 - T128)` |
| Total request time | Request wall time through 256 output tokens; excludes model loading, pack creation, and weight copying |
| Speedup (×) | `engine rate / llama.cpp rate`; llama.cpp = 1.00× |
| Request time saved | `(1 - WICI-InferSSD time / llama.cpp time) × 100%` |

The new Mac first-window metric counts 127 decode intervals between output tokens 1 and 128; TTFT is separate. The historical first-window records below retain their original `128 / T128` convention. Both new and historical README decode values use the final 128 intervals, `128 / (t256 - t128)`. Do not compare the different first-window conventions as if they were identical.

README rates are displayed to three decimal places; full-precision Mac measurements are retained in the run receipt, and padded zeros do not imply additional measurement precision. Speedup ratios for Colibri and WICI-InferSSD use `engine rate / llama.cpp rate`, rounded half up to two decimal places from the supplied summary rates and displayed with the multiplication sign (for example, `3.16×`). The baseline is `1.00×`; ratios below `1.00×` indicate lower throughput. `N/S` means unsupported in the tested version and has no numeric rate or speedup. The ratios may differ slightly from calculations using full-precision records. Different batches, backends, and generated continuations make these comparisons of complete configurations. They do not isolate the benefit of a single optimization or support a pooled speedup across machines.

## Hardware details

The [README hardware table](../README.md#test-hardware) separates system memory, dedicated VRAM, SSD capacity, and SSD interface. Capacities use the nominal device specifications; operating-system readings may be lower and may use GiB instead of GB.

| System | CPU core layout | Model SSD |
| --- | --- | --- |
| MacBook Pro (M5 Max) | 18 cores: 6 super + 12 performance | Apple internal SSD, 2 TB |
| RTX 5060 Ti | 6 cores: 2 performance + 4 efficient; 8 threads | ZHITAI TiPro9000, 2 TB; PCIe 4.0 ×4 |
| RTX 5090 | 10 cores: 6 performance + 4 efficient; 10 threads | ZHITAI TiPro9000, 2 TB; PCIe 5.0 ×4 |
| DGX Spark | 20 cores: 10 Cortex-X925 + 10 Cortex-A725 | Samsung MZALC4T0HBL1, 4 TB; PCIe 5.0 ×4 |

The MacBook Pro is the 16-inch model running macOS 26.4. The TiPro9000 interface entries describe the connections used by the test systems, rather than the maximum interface supported by the SSD model. The Mac's internal storage is listed without an inferred PCIe generation or link width.

The RTX 5060 Ti's 16 GB VRAM and 2 TB SSD, and DGX Spark's 4 TB SSD, were checked against archived setup records on 2026-10-26. CPU details also agree with [Intel Core 3 100U specifications](https://www.intel.com/content/www/us/en/products/sku/238762/intel-core-3-processor-100u-10m-cache-up-to-4-70-ghz-with-ipu/specifications.html), [Intel Core Ultra 5 230F specifications](https://www.intel.com/content/www/us/en/products/sku/241875/intel-core-ultra-5-processor-230f-24m-cache-up-to-5-00-ghz/specifications.html), and [NVIDIA's DGX Spark hardware guide](https://docs.nvidia.com/dgx/dgx-spark/hardware.html).

The RTX 5090 row retains the dataset's **16 GB system-memory configuration**. Archived experiments also include a 32 GB host and a 13.98 GB process-memory limit; installed RAM and a process limit are distinct. Mapping those historical runs to the final summary remains part of the per-run data audit. The hardware metadata does not establish matched RAM budgets across all engines.

Storage probe rates are omitted from the README because they are workload-dependent measurements, not device specifications or expert-loading throughput.

## Engine configurations

**MacBook Pro · current InferSSD (both models).** Measured on 2026-10-26 with the previously validated static Metal binary at `63e165a783f9ee0c8319504b015e5da341e8a8c4`. Its production engine and framework sources are identical to current main `b6a96b30150c9b9945442e506d31b98fd0baec38`; the sole engine difference is a CUDA test file. The binary SHA256 and full-precision results are in [the run receipt](benchmarks/mac-optimized-20261026-4k.json). Each model uses one fresh server and one fixed 4K prompt / 256-token request. No warmup, OS cache purge, parameter sweep or repeated timed request was performed. Background applications were left running.

Both models use internal GGUF files directly, context 8192, batch/ubatch 512/512, 12 prefill threads, CPU routed experts, Metal dense layers, depth-2 reads and 4096-byte read alignment. DeepSeek uses 28 GiB LFU cache, 4 decode threads and polling 0; Muse uses 12 GiB LFU cache, 8 decode threads and polling 50. This tests the measured static Mac presets, not online adaptive control. Model shard metadata and binary hashes are unchanged before/after; both servers exit normally.

**MacBook Pro · DeepSeek baseline.** The historical baseline is `vcruz305/llama.cpp` at `5210c7c5ed61dddaee6ed476623abf4b63093d16`, with default batch/ubatch 2048/512 and 8/12 decode/prefill threads. It uses the same GGUF and source prompt text as the new run. The cache and execution configuration differ. The prior InferSSD 8-GiB-cache, 4096/4096-batch result is preserved in [historical Mac evidence](benchmarks/performance-20261020-4k.json), but is no longer the README's InferSSD row.

**MacBook Pro · Colibri.** Commit `bf2442915d6e3dd4cdfd2eb9c2a3d2aa44a25850`, official safetensors revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, native CPU backend, 8 threads, and 8 cached experts per layer (about **5.60 GiB**). DSpark and Vulkan are disabled. A benchmark patch records token ordinals/timestamps and suppresses EOS without changing compute kernels. Input token IDs matched the historical GGUF runs; precision, backend, cache, batching, and prefill threads differ. The completed run records `passed=true` and describes this CPU configuration.

**MacBook Pro · Muse baseline.** The historical llama.cpp run has different loading behavior, templates and input token counts. Its retained prefill rate is the tester-supplied **3.140** (the archived run rounds to **3.142**). Its speedup ratios describe recorded configurations rather than a matched-input experiment.

**Linux / NVIDIA.** The tester-supplied summary provides 4K prefill and decode rates for all three engines on DeepSeek and for llama.cpp / WICI-InferSSD on Muse. On 2026-10-25, the tester confirmed that all three Linux systems generate 256 output tokens and report the last 128-token window. The earlier RTX 5090 thermal observations and Vulkan configuration are historical details; they are not assumed to describe these measurements.

Mac GGUF runs used fresh processes without warmup or OS page-cache purging. Model loading and previous I/O can affect cache state; these are not controlled cold-cache measurements.

## Checkpoint formats

The verified Mac DeepSeek GGUF runs use `mrfakename/DeepSeek-V4.1-Flash-1017-GGUF` revision `1df54e7d1b1e442051ad393ce871870a929f243e`. Its [model card](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-GGUF) identifies the official 1017 release as the source and pins conversion code to `2076412598e99127ea2554467b2b7530856895d3`.

| Component | Official Safetensors | Recorded Q8_0 GGUF |
| --- | --- | --- |
| Routed experts | Native FP4/MXFP4 | MXFP4 codes and scales repacked without requantization |
| FP8 matrices and Engram tables | FP8 with scales | Dequantized and requantized to Q8_0; Engram uses its own scale layout |
| Other tensors | Includes BF16 and F32 | Includes retained BF16 and F32 tensors |
| Vision and MTP/DSpark | Included in the official release | Omitted from this text-model GGUF |

The conversion paths are visible in the pinned [DeepSeek converter](https://github.com/vcruz305/llama.cpp/blob/2076412598e99127ea2554467b2b7530856895d3/conversion/deepseek.py). The Mac loading log reports **748.49B parameters** and **1,046 tensors**: 120 MXFP4, 332 Q8_0, 65 BF16, and 529 F32. The official **763B** label describes the full published checkpoint. These figures use different module coverage; the GGUF is not a lossless copy of the complete official release.

The Mac llama.cpp and WICI-InferSSD runs use the same GGUF. Mac Colibri uses the official mixed-precision Safetensors described above. This is a throughput comparison of those configurations, not a test of equal model quality. Expert pack creation copies the GGUF expert payload without further quantization.

## Decode windows and request time

All rows below use the **4K prompt tier**. Rates are token/s; times exclude model loading. Each MacBook Pro row is a single run.

| Model / system | Engine | Input tokens | First 128 (token/s) | Last 128 (token/s) | Request time (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| DeepSeek / MacBook Pro | llama.cpp | 4104 | 0.892 | 0.873 | 982.29 |
| DeepSeek / MacBook Pro | Colibri CPU | 4104 | 0.448 | 0.437 | 6370.66 |
| DeepSeek / MacBook Pro | WICI-InferSSD (new) | 4104 | 2.275 | 2.704 | 259.40 |
| Muse / MacBook Pro | llama.cpp historical | 4218 | 0.161 | 0.153 | 2972.35 |
| Muse / MacBook Pro | WICI-InferSSD (new) | 4201 | 0.919 | 0.944 | 610.40 |

New InferSSD prefill took **156.22 s** for DeepSeek and **336.61 s** for Muse. Historical DeepSeek prefill took 692.14 s with llama.cpp and 5792.54 s with Colibri CPU.

## Reproducing a result

Keep the checkpoint revision, quantization, prompt, and template fixed. Record:

1. Source commit, build options, binary hash, command, and relevant environment variables.
2. CPU/GPU, RAM/VRAM, SSD model and link, temperature, power settings, and background I/O.
3. Actual input token IDs, context capacity, batch/ubatch, threads, cache budgets, OS cache-purge policy, and prompt reuse settings.
4. Prefill duration, timestamps for all 256 output tokens, both decode windows, total request time, and model loading time separately.
5. Individual run artifacts, run count, aggregation method, and a correctness check. List failures and timeouts separately from completed measurements.

Use the versioned [prompt text](../bench/prompts/README.md) and [`bench/final_chat.py`](../bench/final_chat.py) against a freshly started server:

```sh
python3 bench/final_chat.py --url http://127.0.0.1:8080 --tier 4096 --output results/model-4k.json --timeout 3600
```

The client records rendered token IDs, raw SSE, per-token timestamps, pre-sampling probabilities and native timing counters. It requires exactly 256 output tokens, a limit stop and no truncation. Server startup, model loading and index creation are outside the timed request. See the [current launcher](../framework/README.md) for deployment.

For an unfinished prefill, `input_tokens / elapsed_prefill` is an **upper bound** on eventual average throughput when the timing interval is correct. It does not reveal processed-token progress or provide a reliable completion estimate.

## Data provenance

The README presentation and historical comparison data were imported from [`wici-ai/ssd-llama` at `a755355f82bc`](https://github.com/wici-ai/ssd-llama/tree/a755355f82bcc52869dd3c1085db59210ced068b). On 2026-10-26 the tester updated RTX 5090 DeepSeek llama.cpp to **8.06 / 0.867**, Colibri to **39.21 / 1.878**, and Muse llama.cpp to **4.0 / 0.155** (prefill / final-128 decode token/s). These are supplied summaries; no baseline remeasurement is claimed.

The new Mac InferSSD measurements use the fixed prompt SHA256 `7eac0ca46f379476846f8221e2ab9617d3023c05f3655769eed523a101754dec`, byte-identical to the historical DeepSeek 4K request's user text. Rendered token counts and template hashes can still differ; exact rendered hashes are recorded in the new receipt. Other platforms retain the source README's InferSSD results. This update does not establish new current-main results for every device or prove no performance regression.

Machine-readable data:

- [Current comparison matrix](benchmarks/reported-comparison-20261026-4k.json).
- [Recalculated speedups](benchmarks/baseline-comparison-20261026-4k.json).
- [Optimized Mac run receipt and all 256 token timestamps](benchmarks/mac-optimized-20261026-4k.json).
- [Historical Mac runs](benchmarks/performance-20261020-4k.json).

The comparison data retains the historical engine key `SSD-Llama`, displayed as **WICI-InferSSD** in the README. Single runs and different engine configurations limit interpretation; speedups do not isolate any one optimization.
