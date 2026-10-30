<h1 align="center">Large MoE models, powered by your SSD</h1>

<p align="center">
  <a href="https://wici.ai">
    <img src="docs/assets/wicissdinfer.png" width="600" alt="InferSSD">
  </a>
</p>

<p align="center">
  <a href="https://wici.ai/inferssd"><b>Website</b></a> &nbsp;·&nbsp;
  <a href="https://wici.ai/research"><b>Blog</b></a> &nbsp;·&nbsp;
  <a href="LICENSE">Apache-2.0</a>
</p>
<p align="center">macOS / Metal &nbsp;·&nbsp; Linux / CUDA &nbsp;·&nbsp; Windows (in progress)</p>

**Run models larger than your RAM or VRAM.** InferSSD builds on [llama.cpp](https://github.com/ggml-org/llama.cpp) to run **DeepSeek-V4.1-Flash-1017** and **Muse-Spark-Open** from SSD. Expert packs, bounded caches, and prompt batching reduce the cost of accessing weights while retaining GGUF, browser chat, and an OpenAI-compatible API.

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

| System | Stage · 4K | llama.cpp<br>Baseline · token/s | Colibri<br>token/s · Speedup | 🟩 **InferSSD**<br>**token/s · Speedup** |
| --- | --- | ---: | ---: | ---: |
| MacBook Pro (M5 Max) | Prefill | 5.929 | 0.708<br>(0.12×) | **26.270**<br>**(4.43×)** |
| | Decode | 0.873 | 0.437<br>(0.50×) | **2.704**<br>**(3.10×)** |
| RTX 5060 Ti | Prefill | 6.950 | 7.020<br>(1.01×) | **75.100**<br>**(10.81×)** |
| | Decode | 0.242 | 0.900<br>(3.72×) | **2.550**<br>**(10.54×)** |
| RTX 5090 | Prefill | 8.060 | 39.210<br>(4.86×) | **164.500**<br>**(20.41×)** |
| | Decode | 0.867 | 1.878<br>(2.17×) | **7.270**<br>**(8.39×)** |
| DGX Spark | Prefill | 2.730 | 0.826<br>(0.30×) | **127.500**<br>**(46.70×)** |
| | Decode | 1.608 | 0.356<br>(0.22×) | **7.260**<br>**(4.51×)** |

### Muse-Spark-Open · 816B

| System | Stage · 4K | llama.cpp<br>Baseline · token/s | Colibri<br>token/s · Speedup | 🟩 **InferSSD**<br>**token/s · Speedup** |
| --- | --- | ---: | ---: | ---: |
| MacBook Pro (M5 Max)<sup>1</sup> | Prefill | 3.140 | N/S | **12.480**<br>**(3.97×)** |
| | Decode | 0.153 | N/S | **0.944**<br>**(6.17×)** |
| RTX 5060 Ti | Prefill | 3.410 | N/S | **52.100**<br>**(15.28×)** |
| | Decode | 0.059 | N/S | **0.219**<br>**(3.71×)** |
| RTX 5090 | Prefill | 4.000 | N/S | **106.600**<br>**(26.65×)** |
| | Decode | 0.155 | N/S | **1.520**<br>**(9.81×)** |
| DGX Spark | Prefill | 1.740 | N/S | **83.900**<br>**(48.22×)** |
| | Decode | 0.120 | N/S | **2.140**<br>**(17.83×)** |

**N/S:** Muse was not supported by the tested Colibri version. It is not a zero-throughput measurement.

<sup>1</sup> The MacBook Pro Muse row uses a historical llama.cpp baseline with a different template and input token count. Its speedup ratios describe those recorded configurations, not a matched-input comparison.

## Get started

Install on Apple Silicon macOS or on Linux with an NVIDIA GPU:

```sh
curl -LsSf https://github.com/wici-ai/inferssd/releases/latest/download/install.sh | sh
```

The installer builds the release from source in a private environment and uses no `sudo`. It needs Git, Python 3.9 or later, and a C/C++ compiler. macOS also needs the Xcode Command Line Tools; Linux also needs the CUDA toolkit, `liburing` headers, and systemd with cgroup v2. Follow the printed PATH instruction if it shows one.

Then start a model:

```sh
# Chat in your terminal
inferssd cli -hf mrfakename/DeepSeek-V4.1-Flash-1017-GGUF

# Or start the web UI and an OpenAI-compatible API at http://127.0.0.1:8080
inferssd serve -hf mrfakename/DeepSeek-V4.1-Flash-1017-GGUF
```

For Muse, use `unsloth/Muse-Spark-Open-816B-A42B-GGUF`.

The first run downloads about 470 GiB of model files to `~/.local/share/inferssd/models` and measures your machine. Use `--model-dir /path/on/ssd` to choose the drive, or `-m /path/to/first-shard.gguf` for files you already have. There is no memory setting to choose: InferSSD uses the memory available. Stop with **Ctrl+C**, or `/exit` in chat.

See the [quick start](docs/quickstart.md#one-command-installation) for upgrades and uninstalling.

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
