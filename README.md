<p align="center">
  <a href="https://wici.ai">
    <img src="docs/assets/wicissdinfer.png" width="600" alt="InferSSD">
  </a>
</p>

<p align="center"><b>Large MoE models, powered by your SSD.</b></p>
<p align="center">macOS / Metal &nbsp;·&nbsp; Linux / CUDA &nbsp;·&nbsp; <a href="LICENSE">Apache-2.0</a></p>

<p align="center">
  <a href="#performance-benchmarks"><b>Benchmarks</b></a> &nbsp;·&nbsp;
  <a href="#test-hardware">Hardware</a> &nbsp;·&nbsp;
  <a href="#get-started"><b>Get started</b></a> &nbsp;·&nbsp;
  <a href="#models">Models</a> &nbsp;·&nbsp;
  <a href="docs/architecture.md">How it works</a> &nbsp;·&nbsp;
  <a href="docs/benchmark.md">Methodology</a>
</p>

**Run models larger than your RAM or VRAM.** InferSSD builds on [llama.cpp](https://github.com/ggml-org/llama.cpp) to run **DeepSeek-V4.1-Flash-1017** and **Muse-Spark-Open** from SSD. Expert packs, bounded caches, and prompt batching reduce the cost of accessing weights while retaining GGUF, browser chat, and an OpenAI-compatible API.

## Model-specific configuration

The typed settings registry now accepts default-empty `model.optimizations`
rules. Native contexts match an exact loaded GGUF name or UUID, architecture, file type,
quantization version and runtime hardware matches into the existing Engram and
hyper-connection graph options. Unknown, unmatched and disabled rules preserve
existing settings and launch defaults; explicit per-option CLI values win.
See [configuration syntax and limits](docs/configuration.md#exact-model-optimization-rules).
Matching happens once per context before graph construction, using metadata
already parsed by the loader. It never hashes or rereads weight payloads. Empty
rules take the original direct-option path with no model-rule logging; configured
rules emit one startup receipt. The previous unmerged `model.content_manifest`
setting and content-hash selector are removed. Weight integrity checks belong to
download/experiment preparation; metadata matching does not certify an artifact.

No production optimization rule is enabled or newly hardware validated. The
native CPU build, UUID-free/name/quantization/device fallback, CLI precedence,
resolver/context ownership tests and typed Python/native
configuration and launch tests use synthetic fixtures; they establish no speedup.
Allocator/THP optimizations require a separate context-specific gated integration.
The historical benchmark bridge recognizes the new installed registry while
retaining its explicit 14/16-setting bindings. It does not drop or translate model
rules into legacy evidence; a binding containing rules still requires separate
reviewed support. Full bench/framework tests cover this compatibility boundary.

## Performance benchmarks

**4K prompt tier · 256 output tokens · decode measured over tokens 129–256.** The 4K label describes the prompt tier, not the model's context-window capacity; actual input token counts depend on the chat template.

All rates are in **token/s**, with higher values indicating faster inference. **llama.cpp is the baseline (1.00×)**. **Speedup** below each Colibri and **InferSSD** rate is `engine throughput / llama.cpp throughput` on the same system. Values above 1.00× are faster; values below 1.00× are slower.

These results compare complete inference configurations. In the Mac DeepSeek runs, Colibri uses official Safetensors on CPU, while the GGUF engines use Metal with CPU experts. See [measurement details](docs/benchmark.md) for the scope of the comparison.

### Test hardware

| Specification | MacBook Pro (M5 Max) | RTX 5060 Ti | RTX 5090 | DGX Spark |
| --- | --- | --- | --- | --- |
| **CPU / SoC** | Apple M5 Max | Intel Core 3 100U | Intel Core Ultra 5 230F | NVIDIA GB10 |
| **CPU cores** | 18 | 6 | 10 | 20 |
| **GPU** | 40-core Apple GPU | RTX 5060 Ti | RTX 5090 | NVIDIA Blackwell |
| **System memory** | 48 GB unified | 8 GB RAM | 32 GB RAM | 128 GB unified |
| **GPU memory** | Shared | 16 GB VRAM | 32 GB VRAM | Shared |
| **SSD capacity** | 2 TB | 2 TB | 2 TB | 4 TB |
| **SSD interface** | Apple internal storage | PCIe 4.0 ×4 | PCIe 5.0 ×4 | PCIe 5.0 ×4 |

Unified memory is shared by the CPU and GPU; dedicated VRAM is additional to system RAM. SSD capacities are nominal. See [hardware details](docs/benchmark.md#hardware-details) for SSD models, CPU core layouts, and configuration notes.

### DeepSeek-V4.1-Flash-1017

#### Prefill · 4K

| System | llama.cpp<br>Baseline · token/s | Colibri<br>token/s · Speedup | 🟩 **InferSSD**<br>**token/s · Speedup** |
| --- | ---: | ---: | ---: |
| MacBook Pro (M5 Max) | 5.929 | 0.708<br>(0.12×) | **26.270**<br>**(4.43×)** |
| RTX 5060 Ti | 6.950 | 7.020<br>(1.01×) | **75.100**<br>**(10.81×)** |
| RTX 5090 | 8.060 | 39.210<br>(4.86×) | **164.500**<br>**(20.41×)** |
| DGX Spark | 2.730 | 0.826<br>(0.30×) | **127.500**<br>**(46.70×)** |

#### Decode · 4K

| System | llama.cpp<br>Baseline · token/s | Colibri<br>token/s · Speedup | 🟩 **InferSSD**<br>**token/s · Speedup** |
| --- | ---: | ---: | ---: |
| MacBook Pro (M5 Max) | 0.873 | 0.437<br>(0.50×) | **2.704**<br>**(3.10×)** |
| RTX 5060 Ti | 0.242 | 0.900<br>(3.72×) | **2.550**<br>**(10.54×)** |
| RTX 5090 | 0.867 | 1.878<br>(2.17×) | **7.270**<br>**(8.39×)** |
| DGX Spark | 1.608 | 0.356<br>(0.22×) | **7.260**<br>**(4.51×)** |

### Muse-Spark-Open · 816B

#### Prefill · 4K

| System | llama.cpp<br>Baseline · token/s | Colibri<br>token/s · Speedup | 🟩 **InferSSD**<br>**token/s · Speedup** |
| --- | ---: | ---: | ---: |
| MacBook Pro (M5 Max)<sup>1</sup> | 3.140 | N/S | **12.480**<br>**(3.97×)** |
| RTX 5060 Ti | 3.410 | N/S | **52.100**<br>**(15.28×)** |
| RTX 5090 | 4.000 | N/S | **106.600**<br>**(26.65×)** |
| DGX Spark | 1.740 | N/S | **83.900**<br>**(48.22×)** |

#### Decode · 4K

| System | llama.cpp<br>Baseline · token/s | Colibri<br>token/s · Speedup | 🟩 **InferSSD**<br>**token/s · Speedup** |
| --- | ---: | ---: | ---: |
| MacBook Pro (M5 Max)<sup>1</sup> | 0.153 | N/S | **0.944**<br>**(6.17×)** |
| RTX 5060 Ti | 0.059 | N/S | **0.219**<br>**(3.71×)** |
| RTX 5090 | 0.155 | N/S | **1.520**<br>**(9.81×)** |
| DGX Spark | 0.120 | N/S | **2.140**<br>**(17.83×)** |

**N/S:** Muse was not supported by the tested Colibri version. It is not a zero-throughput measurement.

<sup>1</sup> The MacBook Pro Muse row uses a historical llama.cpp baseline with a different template and input token count. Its speedup ratios describe those recorded configurations, not a matched-input comparison.

## Get started

Build from `main` for either platform. The setup guide covers source indexing, typed settings, and server launch commands.

| Platform | Branch | Setup |
| --- | --- | --- |
| macOS / Metal (Apple Silicon) | `main` | [Mac setup](docs/quickstart.md#apple-silicon) · [Muse on Mac](docs/quickstart.md#muse-on-mac) |
| Linux / NVIDIA CUDA | `main` | [Linux setup](docs/quickstart.md#linux--nvidia) |

`x86_64` covers Intel and AMD CPUs; DGX Spark (GB10) uses `ARM64`. CUDA identifies the NVIDIA GPU backend, not the CPU vendor. The x86_64 benchmarks use Intel CPUs.

Historical DeepSeek runs used approximately **473 GiB of GGUF weights + 269 GiB for the expert pack**; Muse's pack added approximately **450 GiB**. The current framework can index GGUF experts directly without a separate pack. When using a pack, keep the original shards and generate it from the exact checkpoint you load.

Mac requires Xcode/Metal, CMake, Ninja, and Python. Linux requires a compatible CUDA toolkit, CMake, Python, and `liburing`; its launcher uses systemd/cgroup v2. The Mac example targets **M5 Max / 48 GB**; choose cache and batch budgets that fit your machine.

## Models

These GGUF references are recorded in the verified Mac runs. They are community conversions of the named model releases.

| Model | Parameter count | GGUF reference |
| --- | --- | --- |
| DeepSeek-V4.1-Flash-1017 | 763B official release<br>748.49B in the text GGUF | [mrfakename · Q8_0 with native MXFP4 experts](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-GGUF) |
| Muse-Spark-Open | 816B total / 42B active | [Unsloth · UD-Q4_K_XL](https://huggingface.co/unsloth/Muse-Spark-Open-816B-A42B-GGUF) |

DeepSeek's **Q8_0 label does not mean every tensor is 8-bit**: routed experts retain MXFP4, while other tensors include Q8_0, BF16, and F32. This text GGUF omits the vision and MTP/DSpark modules. Expert packing copies the existing quantized expert bytes without requantization. See [weight-format details](docs/benchmark.md#checkpoint-formats).

## Use it with your tools

Once the server is ready, open its address in your browser or point an OpenAI-compatible client at its `/v1` endpoint.

| Setup | Browser | API base URL |
| --- | --- | --- |
| Quick start | `http://127.0.0.1:8080` | `http://127.0.0.1:8080/v1` |

Send a request to the default port `8080`:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Explain how a mixture-of-experts model works."}],
    "max_tokens": 256,
    "temperature": 0,
    "stream": true,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

## Further reading

- [How it works](docs/architecture.md): expert packs, caching, batching, and the module map.
- [Engine configurations](docs/benchmark.md#engine-configurations) and [decode windows / request times](docs/benchmark.md#decode-windows-and-request-time).
- [Reproducing a result](docs/benchmark.md#reproducing-a-result) and [4K data provenance](docs/benchmark.md#data-provenance).

## Contributing

Hardware results and improvements to expert I/O, caching, and batching are welcome. Include the commit, checkpoint, hardware, command, cache state, actual prompt length, raw timings, and a correctness check. Use the [measurement checklist](docs/benchmark.md#reproducing-a-result).

## Acknowledgments and license

Built on [llama.cpp and ggml](https://github.com/ggml-org/llama.cpp), with thanks to their contributors and the model-adaptation community.

InferSSD is licensed under [Apache-2.0](LICENSE); bundled llama.cpp/ggml and other third-party components keep their original licenses and notices; see [NOTICE](NOTICE). Model weights remain subject to their publishers' licenses.
