<!-- SPDX-License-Identifier: Apache-2.0 -->
# Benchmark methodology

[Back to 4K results](../README.md#performance-benchmarks)

The current README summary is available as [machine-readable data](benchmarks/current-comparison.json). The dated snapshots and detailed run timings below describe earlier measurements, before the latest Mac and RTX 5090 updates. They are retained as historical evidence and do not replace the current README values.

The README comparisons use the **4K prompt tier**. This is the input package's label, not the context capacity or an exact token count: chat templates add tokens. All displayed decode rates use **tokens 129–256 of a 256-token output**. The tester confirmed this convention for RTX 5060 Ti, RTX 5090, and DGX Spark on **2026-10-25**. Mac runs use an **8192-token context**, one request, greedy sampling, no prompt-cache reuse, and EOS suppression. Device specifications are listed in [Test hardware](../README.md#test-hardware).

## Measurement definitions

| Metric | Definition |
| --- | --- |
| Prefill throughput | Actual prompt tokens / prompt-processing seconds |
| First 128 output tokens | `128 / T128` |
| Last 128 output tokens | `128 / (T256 - T128)` |
| Total request time | Request wall time through 256 output tokens; excludes model loading, pack creation, and weight copying |
| Speedup (×) | `engine rate / llama.cpp rate`; llama.cpp = 1.00× |
| Request time saved | `(1 - InferSSD time / llama.cpp time) × 100%` |

`T128` and `T256` are cumulative generation times after prefill. The first output token comes from prefill, so the first window contains 127 subsequent decode intervals; these records consistently use the historical `128 / T128` convention. The README comparison tables use the **last 128-token window on every platform**.

README rates are formatted to three decimal places without changing their numeric values; padded zeros do not imply additional measurement precision. Speedup ratios for Colibri and InferSSD use `engine rate / llama.cpp rate`, rounded half up to two decimal places from the supplied summary rates and displayed with the multiplication sign (for example, `3.16×`). The baseline is `1.00×`; ratios below `1.00×` indicate lower throughput. `N/S` means unsupported in the tested version and has no numeric rate or speedup. The ratios may differ slightly from calculations using full-precision records. Different batches, backends, and generated continuations make these comparisons of complete configurations. They do not isolate the benefit of a single optimization or support a pooled speedup across machines.

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

The RTX 5090 host has **32 GB of system memory**. Archived experiments also include a 13.98 GB process-memory limit; installed RAM and a process limit are distinct. Those archived runs are not mapped to individual summary values, and the summary does not establish matched RAM budgets across all engines.

Storage probe rates are omitted from the README because they are workload-dependent measurements, not device specifications or expert-loading throughput.

## Engine configurations

The configurations in this section describe the archived benchmark runs. Current deployment uses the typed interface in [Quick start](quickstart.md); the newer README rates are summary values and are not assigned the older runs' timing artifacts.

**MacBook Pro · DeepSeek.** The baseline is the `vcruz305/llama.cpp` DeepSeek-support fork at `5210c7c5ed61dddaee6ed476623abf4b63093d16`. Its batch flags were omitted, giving defaults **2048/512**. InferSSD was tested at predecessor-codebase revision `8010461f77190413416387ce50b86532fad2fdb4` (merged as `aee25da468cbdbb7c72d2293f86262d6933da657`), using **4096/4096**, adaptation off, and an **8 GiB LFU** expert cache. Both use the same GGUF, prompt, template, 8/12 decode/prefill threads, Metal dense computation, and CPU routed experts. Each 4K result is one run. The baseline uses default batching; its context, threads, placement, and sampling are explicitly configured.

**MacBook Pro · Colibri.** [Colibri](https://github.com/JustVugg/colibri) at commit `bf2442915d6e3dd4cdfd2eb9c2a3d2aa44a25850`, official safetensors revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, native CPU backend, 8 threads, and 8 cached experts per layer (about **5.60 GiB**). DSpark and Vulkan are disabled. A benchmark patch records token ordinals/timestamps and suppresses EOS without changing compute kernels. Input token IDs match the GGUF runs; precision, backend, cache, batching, and prefill threads differ. The completed run records `passed=true` and describes this CPU configuration.

**MacBook Pro · Muse.** InferSSD uses the Mac revision above, an 8 GiB LFU cache, compact Metal allocations, and `muse-spark.jinja`. The historical llama.cpp run has different loading behavior, templates, and input token counts. The main table retains the supplied baseline prefill rate **3.140**; the archived run rounds to **3.142**. Its marked speedup ratios are historical configuration comparisons.

**Linux / NVIDIA.** The tester-supplied summary provides 4K prefill and decode rates for all three engines on DeepSeek and for llama.cpp / InferSSD on Muse. On 2026-10-25, the tester confirmed that all three Linux systems generate 256 output tokens and report the last 128-token window. The earlier RTX 5090 thermal observations and Vulkan configuration are historical details; they are not assumed to describe these measurements.

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

The Mac llama.cpp and InferSSD runs use the same GGUF. Mac Colibri uses the official mixed-precision Safetensors described above. This is a throughput comparison of those configurations, not a test of equal model quality. Expert pack creation copies the GGUF expert payload without further quantization.

## Decode windows and request time

All rows below use the **4K prompt tier**. Rates are token/s; times exclude model loading. Each MacBook Pro row is one archived run. These older timings do not correspond to the newer InferSSD rates in the current README.

| Model / system | Engine | Input tokens | First 128 (token/s) | Last 128 (token/s) | Request time (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| DeepSeek / MacBook Pro | llama.cpp | 4104 | 0.892 | 0.873 | 982.29 |
| DeepSeek / MacBook Pro | Colibri CPU | 4104 | 0.448 | 0.437 | 6370.66 |
| DeepSeek / MacBook Pro | InferSSD | 4104 | 1.973 | 2.039 | 346.67 |
| Muse / MacBook Pro | llama.cpp historical | 4218 | 0.161 | 0.153 | 2972.35 |
| Muse / MacBook Pro | InferSSD | 4201 | 0.766 | 0.766 | 862.14 |

For MacBook Pro DeepSeek, prefill took **692.14 s** with llama.cpp, **218.96 s** with InferSSD, and **5792.54 s** with Colibri CPU. Colibri's complete request took **1 h 46 min 10.66 s**.

## Reproducing a result

Keep the checkpoint revision, quantization, prompt, and template fixed. Record:

1. Source commit, build options, binary hash, command, and relevant environment variables.
2. CPU/GPU, RAM/VRAM, SSD model and link, temperature, power settings, and background I/O.
3. Actual input token IDs, context capacity, batch/ubatch, threads, cache budgets, OS cache-purge policy, and prompt reuse settings.
4. Prefill duration, timestamps for all 256 output tokens, both decode windows, total request time, and model loading time separately.
5. Individual run artifacts, run count, aggregation method, and a correctness check. List failures and timeouts separately from completed measurements.

Use the current [benchmark tools](../bench/README.md) for measurements. Keep deployment checks separate from formal timing and numerical evidence.

For an unfinished prefill, `input_tokens / elapsed_prefill` is an **upper bound** on eventual average throughput when the timing interval is correct. It does not reveal processed-token progress or provide a reliable completion estimate.

## Data provenance

The Mac results were checked against saved configurations and summaries; the completed Colibri run was retrieved from the test machine. The dated comparison snapshots use the tester-supplied **4K** matrix recorded on **2026-10-22**; the current README includes subsequent Mac and RTX 5090 updates. The RTX 5090 CPU is corrected to **Core Ultra 5 230F**, and its SSD interface to **PCIe Gen5 ×4**; the previously supplied 32 GB VRAM and 2 TB capacity are retained. Linux rates remain tester-reported summaries; confirming the decode definition does not constitute a new timing measurement. The hardware review on **2026-10-26** fills the RTX 5060 Ti VRAM and SSD fields and the DGX Spark SSD fields from archived setup records; see [hardware details](#hardware-details). The tester confirmed on the same date that the current Muse baseline values supersede earlier incomplete runs and use the stated final 128-token window.

The MacBook Pro DeepSeek inputs are identified by:

- **Prompt ZIP SHA256:** `22dd769cf8bbcf599efcd8b72195c89f21dc33d320323856d1eed2a90f59a1fe`
- **Chat template SHA256:** `f6b5ab27e4bd357866487dd71a5026affe1d272c2d6411bcf82fb5937b00abff`

Historical machine-readable **4K-only** data: [verified Mac runs](benchmarks/performance-20261020-4k.json), [comparison data](benchmarks/reported-comparison-20261022-4k.json), and [baseline calculations](benchmarks/baseline-comparison-20261022-4k.json). These snapshots cover the four platforms shown in the README. The prompt package identified above is not distributed with this repository; new measurements should use the fixed prompts in [bench/prompts](../bench/prompts/README.md). The [setup commands](quickstart.md) are deployment examples.
