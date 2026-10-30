<h1 align="center">Large MoE models, powered by your SSD</h1>

<p align="center">
  <a href="https://wici.ai">
    <img src="docs/assets/wicissdinfer.png" width="600" alt="InferSSD">
  </a>
</p>

<p align="center">
  <a href="https://wici.ai/inferssd"><b>Website</b></a> &nbsp;·&nbsp;
  <a href="https://wici.ai/research"><b>Blog</b></a> &nbsp;·&nbsp;
  <a href="https://join.slack.com/t/wici-ai/shared_invite/zt-4f2e7ob2s-9nUg35Phqyqt4EI6qUjw5w"><b>Slack</b></a> &nbsp;·&nbsp;
  <a href="LICENSE">Apache-2.0</a>
</p>
<p align="center">macOS / Metal &nbsp;·&nbsp; Linux / CUDA &nbsp;·&nbsp; Windows (in progress)</p>

**Run models larger than your RAM or VRAM.** InferSSD builds on [llama.cpp](https://github.com/ggml-org/llama.cpp) to run **DeepSeek-V4.1-Flash-1017** and **Muse-Spark-Open** from SSD. Expert packs, bounded caches, and prompt batching reduce the cost of accessing weights while retaining GGUF, browser chat, and an OpenAI-compatible API.

## Performance benchmarks

**4K prompt tier · 256 output tokens · decode measured over tokens 129–256.** The 4K label describes the prompt tier, not the model's context-window capacity; actual input token counts depend on the chat template.

All rates are in **token/s**, with higher values indicating faster inference. **llama.cpp is the baseline (1.00×)**. The multiple next to each **InferSSD** rate is `engine throughput / llama.cpp throughput` on the same system. Values above 1.00× are faster; values below 1.00× are slower.

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

<table>
<thead>
<tr><th rowspan="2">System</th><th colspan="3">Prefill (token/s)</th><th colspan="3">Decode (token/s)</th></tr>
<tr><th>llama.cpp</th><th>Colibri</th><th>InferSSD</th><th>llama.cpp</th><th>Colibri</th><th>InferSSD</th></tr>
</thead>
<tbody>
<tr><td>MacBook&nbsp;Pro&nbsp;M5&nbsp;Max</td><td align="right">5.929</td><td align="right">0.708</td><td align="right"><b>26.270</b>&nbsp;(4.43×)</td><td align="right">0.873</td><td align="right">0.437</td><td align="right"><b>2.704</b>&nbsp;(3.10×)</td></tr>
<tr><td>RTX 5060 Ti</td><td align="right">6.950</td><td align="right">7.020</td><td align="right"><b>75.100</b>&nbsp;(10.81×)</td><td align="right">0.242</td><td align="right">0.900</td><td align="right"><b>2.550</b>&nbsp;(10.54×)</td></tr>
<tr><td>RTX 5090</td><td align="right">8.060</td><td align="right">39.210</td><td align="right"><b>164.500</b>&nbsp;(20.41×)</td><td align="right">0.867</td><td align="right">1.878</td><td align="right"><b>7.270</b>&nbsp;(8.39×)</td></tr>
<tr><td>DGX Spark</td><td align="right">2.730</td><td align="right">0.826</td><td align="right"><b>127.500</b>&nbsp;(46.70×)</td><td align="right">1.608</td><td align="right">0.356</td><td align="right"><b>7.260</b>&nbsp;(4.51×)</td></tr>
</tbody>
</table>

### Muse-Spark-Open · 816B

<table>
<thead>
<tr><th rowspan="2">System</th><th colspan="3">Prefill (token/s)</th><th colspan="3">Decode (token/s)</th></tr>
<tr><th>llama.cpp</th><th>Colibri</th><th>InferSSD</th><th>llama.cpp</th><th>Colibri</th><th>InferSSD</th></tr>
</thead>
<tbody>
<tr><td>MacBook&nbsp;Pro&nbsp;M5&nbsp;Max<sup>1</sup></td><td align="right">3.140</td><td align="right">N/S</td><td align="right"><b>12.480</b>&nbsp;(3.97×)</td><td align="right">0.153</td><td align="right">N/S</td><td align="right"><b>0.944</b>&nbsp;(6.17×)</td></tr>
<tr><td>RTX 5060 Ti</td><td align="right">3.410</td><td align="right">N/S</td><td align="right"><b>52.100</b>&nbsp;(15.28×)</td><td align="right">0.059</td><td align="right">N/S</td><td align="right"><b>0.219</b>&nbsp;(3.71×)</td></tr>
<tr><td>RTX 5090</td><td align="right">4.000</td><td align="right">N/S</td><td align="right"><b>106.600</b>&nbsp;(26.65×)</td><td align="right">0.155</td><td align="right">N/S</td><td align="right"><b>1.520</b>&nbsp;(9.81×)</td></tr>
<tr><td>DGX Spark</td><td align="right">1.740</td><td align="right">N/S</td><td align="right"><b>83.900</b>&nbsp;(48.22×)</td><td align="right">0.120</td><td align="right">N/S</td><td align="right"><b>2.140</b>&nbsp;(17.83×)</td></tr>
</tbody>
</table>

**N/S:** Muse was not supported by the tested Colibri version. It is not a zero-throughput measurement.

<sup>1</sup> The MacBook Pro Muse row uses a historical llama.cpp baseline with a different template and input token count. Its speedup ratios describe those recorded configurations, not a matched-input comparison.

## Get started

Install on Apple Silicon macOS or on Linux with an NVIDIA GPU:

```sh
curl -fsSL https://github.com/wici-ai/inferssd/releases/latest/download/install.sh | sh
```

InferSSD currently installs from source, so the engine is compiled for your machine's CPU and GPU; prebuilt packages are not available yet. Building takes about 2 to 12 minutes, depending on the machine.

The installer works in a private environment and uses no `sudo`. It needs Git, Python 3.9 or later, and a C/C++ compiler. macOS also needs the Xcode Command Line Tools; Linux also needs the CUDA toolkit, `liburing` headers, and systemd with cgroup v2. Follow the printed PATH instruction if it shows one.

Then start a model:

```sh
# Chat in your terminal
inferssd cli -hf mrfakename/DeepSeek-V4.1-Flash-1017-GGUF

# Or start the web UI and an OpenAI-compatible API at http://127.0.0.1:8080
inferssd serve -hf mrfakename/DeepSeek-V4.1-Flash-1017-GGUF
```

For Muse, use `unsloth/Muse-Spark-Open-816B-A42B-GGUF`. With `serve`, open `http://127.0.0.1:8080` in your browser, or point any OpenAI-compatible client at `http://127.0.0.1:8080/v1`.

The first run downloads about 470 GiB of model files to `~/.local/share/inferssd/models` and measures your machine. Use `--model-dir /path/on/ssd` to choose the drive, or `-m /path/to/first-shard.gguf` for files you already have. There is no memory setting to choose: InferSSD uses the memory available. Stop with **Ctrl+C**, or `/exit` in chat.

See the [quick start](docs/quickstart.md#one-command-installation) for upgrades and uninstalling.

## Models

These GGUF references are recorded in the verified Mac runs. They are community conversions of the named model releases.

| Model | Parameter count | GGUF reference |
| --- | --- | --- |
| DeepSeek-V4.1-Flash-1017 | 763B official release<br>748.49B in the text GGUF | [mrfakename · Q8_0 with native MXFP4 experts](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-GGUF) |
| Muse-Spark-Open | 816B total / 42B active | [Unsloth · UD-Q4_K_XL](https://huggingface.co/unsloth/Muse-Spark-Open-816B-A42B-GGUF) |

DeepSeek's **Q8_0 label does not mean every tensor is 8-bit**: routed experts retain MXFP4, while other tensors include Q8_0, BF16, and F32. This text GGUF omits the vision and MTP/DSpark modules. Expert packing copies the existing quantized expert bytes without requantization. See [weight-format details](docs/benchmark.md#checkpoint-formats).

## Further reading

- [How it works](docs/architecture.md): expert packs, caching, batching, and the module map.
- [Engine configurations](docs/benchmark.md#engine-configurations) and [decode windows / request times](docs/benchmark.md#decode-windows-and-request-time).
- [Reproducing a result](docs/benchmark.md#reproducing-a-result) and [4K data provenance](docs/benchmark.md#data-provenance).

## Contributing

Contributions are welcome, including hardware results and improvements to expert I/O, caching, and batching. See the [contributing guide](CONTRIBUTING.md).

## Acknowledgments and license

Built on [llama.cpp and ggml](https://github.com/ggml-org/llama.cpp), with thanks to their contributors and the model-adaptation community.

InferSSD is licensed under [Apache-2.0](LICENSE); bundled llama.cpp/ggml and other third-party components keep their original licenses and notices; see [NOTICE](NOTICE). Model weights remain subject to their publishers' licenses.
