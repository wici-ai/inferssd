<h1 align="center">Large MoE models, powered by your SSD</h1>

<p align="center">
  <a href="https://wici.ai">
    <img src="docs/assets/wicissdinfer.png" width="600" alt="InferSSD">
  </a>
</p>

<p align="center">
  <a href="https://wici.ai/wici-infer#app-download"><b>Download app</b></a> &nbsp;·&nbsp;
  <a href="https://wici.ai/inferssd"><b>Website</b></a> &nbsp;·&nbsp;
  <a href="https://wici.ai/research"><b>Blog</b></a> &nbsp;·&nbsp;
  <a href="https://discord.gg/wici"><b>Discord</b></a> &nbsp;·&nbsp;
  <a href="https://join.slack.com/t/wici-ai/shared_invite/zt-4f2e7ob2s-9nUg35Phqyqt4EI6qUjw5w"><b>Slack</b></a> &nbsp;·&nbsp;
  <a href="LICENSE">Apache-2.0</a> &nbsp;·&nbsp;
  <a href="docs/README.zh-CN.md">简体中文</a>
</p>
<p align="center">macOS (Metal) &nbsp;·&nbsp; Windows 10/11 (CUDA · Vulkan · ROCm in progress) &nbsp;·&nbsp; Linux (CUDA · ROCm · Vulkan in progress)</p>

**Run models larger than your RAM or VRAM.** InferSSD builds on [llama.cpp](https://github.com/ggml-org/llama.cpp) to run large mixture-of-experts (MoE) models from SSD, using standard GGUF files, with browser chat and an OpenAI-compatible API. It runs **DeepSeek-V4.1-Flash-1017** and **Qwen3.8 Flash Next** in many quantizations, as well as **Muse-Spark-Open**, **Qwen3.6-35B-A3B** and **Gemma4 E4B**.

## Models

These are the tested GGUF files, used as their publishers released them. Expand a model for its files.

<details>
<summary><b>DeepSeek-V4.1-Flash-1017</b> · 763B total / 8–16B active · 10 GGUFs</summary>

| Quantization | Publisher | Size (GB) |
| --- | --- | ---: |
| [UD-IQ1_S](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/main/UD-IQ1_S) | [Unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) | 233.0 |
| [UD-IQ1_M](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/main/UD-IQ1_M) | [Unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) | 250.4 |
| [UD-IQ2_XXS](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/main/UD-IQ2_XXS) | [Unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) | 300.5 |
| [UD-Q2_K_XL](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/main/UD-Q2_K_XL) | [Unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) | 334.8 |
| [UD-IQ3_XXS](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/main/UD-IQ3_XXS) | [Unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) | 394.7 |
| [UD-IQ3_S](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/main/UD-IQ3_S) | [Unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) | 423.6 |
| [UD-Q3_K_XL](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/main/UD-Q3_K_XL) | [Unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) | 485.1 |
| [UD-Q4_K_XL](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/main/UD-Q4_K_XL) | [Unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) | 500.7 |
| [Q8_0](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-GGUF/tree/main/Q8_0) | [mrfakename](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-GGUF) | 508.0 |
| [UD-Q8_K_XL](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/main/UD-Q8_K_XL) | [Unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) | 508.4 |

mrfakename's Q8_0 keeps the official bit widths: routed experts stay native 4-bit MXFP4, and FP8 tensors become 8-bit Q8_0. See [weight-format details](docs/benchmark.md#checkpoint-formats).

</details>

<details>
<summary><b>Muse-Spark-Open</b> · 816B total / 42B active · 1 GGUF</summary>

| Quantization | Publisher | Size (GB) |
| --- | --- | ---: |
| [UD-Q4_K_XL](https://huggingface.co/unsloth/Muse-Spark-Open-816B-A42B-GGUF/tree/main/UD-Q4_K_XL) | [Unsloth](https://huggingface.co/unsloth/Muse-Spark-Open-816B-A42B-GGUF) | 505.5 |

</details>

<details>
<summary><b>Qwen3.8 Flash Next</b> · 177B total / A3B · 14 GGUFs</summary>

| Quantization | Publisher | Size (GB) |
| --- | --- | ---: |
| [UD-IQ1_S](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/main/UD-IQ1_S) | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 66.4 |
| [Q2_0](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/main/Q2_0) | [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | 66.4 |
| [UD-IQ1_M](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/main/UD-IQ1_M) | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 68.0 |
| [IQ2_XS](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/main/IQ2_XS) | [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | 68.0 |
| [UD-Q2_0](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/main/UD-Q2_0) | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 70.5 |
| [UD-IQ2_XS](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/main/UD-IQ2_XS) | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 75.8 |
| [IQ3_XXS](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/main/IQ3_XXS) | [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | 75.8 |
| [UD-Q2_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/main/UD-Q2_K_XL) | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 79.6 |
| [UD-IQ3_XXS](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/main/UD-IQ3_XXS) | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 83.6 |
| [IQ3_S](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/main/IQ3_S) | [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | 83.6 |
| [UD-Q3_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/main/UD-Q3_K_XL) | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 89.5 |
| [UD-IQ4_XS](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/main/UD-IQ4_XS) | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 93.7 |
| [IQ4_XS](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-IQ4_XS) | [Bartowski](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF) | 97.7 |
| [UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/main/UD-Q4_K_XL) | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 111.3 |

</details>

<details>
<summary><b>Qwen3.6-35B-A3B</b> · 35B total / 3B active · 1 GGUF</summary>

| Quantization | Publisher | Size (GB) |
| --- | --- | ---: |
| [UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/blob/main/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf) | [Unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | 22.4 |

</details>

<details>
<summary><b>Gemma4 E4B</b> · 7.46B total / E4B · 1 GGUF</summary>

| Quantization | Publisher | Size (GB) |
| --- | --- | ---: |
| [QAT Q4_0](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/blob/main/gemma-4-E4B_q4_0-it.gguf) | [Google](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf) | 5.2 |

</details>

## Performance benchmarks

**4K prompt tier · 256 output tokens · decode measured over tokens 129–256.** All rates are in **token/s**; higher is faster. The multiple next to each **InferSSD** rate is its speedup over llama.cpp on the same hardware. On every machine, llama.cpp and InferSSD run DeepSeek's [mrfakename Q8_0](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-GGUF/tree/main/Q8_0) and Muse's [Unsloth UD-Q4_K_XL](https://huggingface.co/unsloth/Muse-Spark-Open-816B-A42B-GGUF/tree/main/UD-Q4_K_XL); on the MacBook Pro they run them with Metal, and Colibri runs the official Safetensors on CPU. See [measurement details](docs/benchmark.md).

### Test hardware

| Hardware | CPU (cores) | GPU | Memory | SSD |
| --- | --- | --- | --- | --- |
| MacBook&nbsp;Pro | Apple&nbsp;M5&nbsp;Max&nbsp;(18) | 40-core | 48&nbsp;GB&nbsp;unified | 2&nbsp;TB&nbsp;internal |
| RTX&nbsp;5060&nbsp;Ti | Intel&nbsp;Core&nbsp;3&nbsp;100U&nbsp;(6) | RTX&nbsp;5060&nbsp;Ti | 8&nbsp;GB&nbsp;+&nbsp;16&nbsp;GB&nbsp;VRAM | 2&nbsp;TB&nbsp;PCIe&nbsp;4&nbsp;×4 |
| RTX&nbsp;5090 | Intel&nbsp;Core&nbsp;Ultra&nbsp;5&nbsp;230F&nbsp;(10) | RTX&nbsp;5090 | 32&nbsp;GB&nbsp;+&nbsp;32&nbsp;GB&nbsp;VRAM | 2&nbsp;TB&nbsp;PCIe&nbsp;5&nbsp;×4 |
| DGX&nbsp;Spark | NVIDIA&nbsp;GB10&nbsp;(20) | Blackwell | 128&nbsp;GB&nbsp;unified | 4&nbsp;TB&nbsp;PCIe&nbsp;5&nbsp;×4 |

Unified memory is shared by the CPU and GPU. SSD capacities are nominal. See [hardware details](docs/benchmark.md#hardware-details) for SSD models, CPU core layouts, and configuration notes.

### DeepSeek-V4.1-Flash-1017 · mrfakename Q8_0

<table>
<thead>
<tr><th rowspan="2">Hardware</th><th colspan="3">Prefill (token/s)</th><th colspan="3">Decode (token/s)</th></tr>
<tr><th>llama.cpp</th><th>Colibri</th><th>InferSSD</th><th>llama.cpp</th><th>Colibri</th><th>InferSSD</th></tr>
</thead>
<tbody>
<tr><td>MacBook&nbsp;Pro</td><td align="right">5.929</td><td align="right">0.708</td><td align="right"><b>28.766</b>&nbsp;(4.85×)</td><td align="right">0.873</td><td align="right">0.437</td><td align="right"><b>2.874</b>&nbsp;(3.29×)</td></tr>
<tr><td>RTX 5060 Ti</td><td align="right">6.950</td><td align="right">7.020</td><td align="right"><b>75.100</b>&nbsp;(10.81×)</td><td align="right">0.242</td><td align="right">0.900</td><td align="right"><b>2.550</b>&nbsp;(10.54×)</td></tr>
<tr><td>RTX 5090</td><td align="right">8.060</td><td align="right">39.210</td><td align="right"><b>164.500</b>&nbsp;(20.41×)</td><td align="right">0.867</td><td align="right">1.878</td><td align="right"><b>7.270</b>&nbsp;(8.39×)</td></tr>
<tr><td>DGX Spark</td><td align="right">2.730</td><td align="right">0.826</td><td align="right"><b>127.500</b>&nbsp;(46.70×)</td><td align="right">1.608</td><td align="right">0.356</td><td align="right"><b>7.260</b>&nbsp;(4.51×)</td></tr>
</tbody>
</table>

### Muse-Spark-Open · Unsloth UD-Q4_K_XL

<table>
<thead>
<tr><th rowspan="2">Hardware</th><th colspan="3">Prefill (token/s)</th><th colspan="3">Decode (token/s)</th></tr>
<tr><th>llama.cpp</th><th>Colibri</th><th>InferSSD</th><th>llama.cpp</th><th>Colibri</th><th>InferSSD</th></tr>
</thead>
<tbody>
<tr><td>MacBook&nbsp;Pro</td><td align="right">3.140</td><td align="right">N/S</td><td align="right"><b>14.503</b>&nbsp;(4.62×)</td><td align="right">0.153</td><td align="right">N/S</td><td align="right"><b>1.097</b>&nbsp;(7.17×)</td></tr>
<tr><td>RTX 5060 Ti</td><td align="right">3.410</td><td align="right">N/S</td><td align="right"><b>52.100</b>&nbsp;(15.28×)</td><td align="right">0.059</td><td align="right">N/S</td><td align="right"><b>0.219</b>&nbsp;(3.71×)</td></tr>
<tr><td>RTX 5090</td><td align="right">4.000</td><td align="right">N/S</td><td align="right"><b>106.600</b>&nbsp;(26.65×)</td><td align="right">0.155</td><td align="right">N/S</td><td align="right"><b>1.520</b>&nbsp;(9.81×)</td></tr>
<tr><td>DGX Spark</td><td align="right">1.740</td><td align="right">N/S</td><td align="right"><b>83.900</b>&nbsp;(48.22×)</td><td align="right">0.120</td><td align="right">N/S</td><td align="right"><b>2.140</b>&nbsp;(17.83×)</td></tr>
</tbody>
</table>

**N/S:** Muse is not currently supported by Colibri.

## Get started

**Desktop app.** [WiCi Infer](https://wici.ai/wici-infer) is a free app for macOS (Apple silicon) and Windows (x64), built on InferSSD. It sets up the engine, downloads models and runs chat, with no terminal. [Download for macOS](https://appapi.wici.ai/versioning/wici-infer/redirect/macos-latest) · [Download for Windows](https://appapi.wici.ai/versioning/wici-infer/redirect/windows-latest)

**Command line.** On macOS or Linux:

```sh
curl -fsSL https://github.com/wici-ai/inferssd/releases/latest/download/install.sh | sh
```

On Windows, in PowerShell:

```powershell
irm https://github.com/wici-ai/inferssd/releases/latest/download/install.ps1 | iex
```

On macOS 15 or later on Apple silicon and on Windows 10 or 11 x64, the installer downloads the release's prebuilt [macOS](docs/macos-package.md) or [Windows](docs/windows-package.md) package, with no developer tools. On Linux it builds the release from source for the machine's CPU and GPU, which takes about 2 to 12 minutes and needs Git, Python 3.9 or later, a C/C++ compiler, systemd with cgroup v2, and the CUDA toolkit or the ROCm HIP SDK for the GPU. The installer works in a private environment and uses no `sudo`. Follow the printed PATH instruction if it shows one.

Then start a model:

```sh
# Chat in your terminal
inferssd cli -hf mrfakename/DeepSeek-V4.1-Flash-1017-GGUF

# Or start the web UI and an OpenAI-compatible API at http://127.0.0.1:8080
inferssd serve -hf mrfakename/DeepSeek-V4.1-Flash-1017-GGUF
```

`-hf` downloads and runs `mrfakename/DeepSeek-V4.1-Flash-1017-GGUF` (Q8_0), `unsloth/Muse-Spark-Open-816B-A42B-GGUF` (UD-Q4_K_XL), `unsloth/Qwen3.6-35B-A3B-GGUF` (UD-Q4_K_XL), `bartowski/Qwen3.8-Flash-Next-GGUF` (IQ4_XS) and `google/gemma-4-E4B-it-qat-q4_0-gguf`. For any other file in [Models](#models), download all of its files to one folder and pass the first with `-m`:

```sh
hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF --include "UD-IQ1_S/*" --local-dir ~/models/deepseek
inferssd serve -m ~/models/deepseek/UD-IQ1_S/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf
```

With `serve`, open `http://127.0.0.1:8080` in your browser, or point any OpenAI-compatible client at `http://127.0.0.1:8080/v1`. Add `--host 0.0.0.0` to reach it from other devices on your network; the API has no authentication, so anyone who can reach the port can use it.

The first `-hf` run downloads the model files to `~/.local/share/inferssd/models` (`%LOCALAPPDATA%\inferssd\models` on Windows) and measures your machine. Use `--model-dir /path/on/ssd` to choose the drive. There is no memory setting to choose: InferSSD uses the memory available. Stop with **Ctrl+C**, or `/exit` in chat.

See the [quick start](docs/quickstart.md#one-command-installation) for upgrades and uninstalling.

## Further reading

- [How it works](docs/architecture.md)
- [Benchmark methodology](docs/benchmark.md)
- [Quick start](docs/quickstart.md)
- [Versions and engine resources](docs/engine-resources.md)

## Contributing

Contributions are welcome, including hardware results and improvements to expert I/O, caching, and batching. See the [contributing guide](CONTRIBUTING.md).

## Acknowledgments and license

Built on [llama.cpp and ggml](https://github.com/ggml-org/llama.cpp), with thanks to their contributors and the model-adaptation community.

InferSSD is licensed under [Apache-2.0](LICENSE); bundled llama.cpp/ggml and other third-party components keep their original licenses and notices; see [NOTICE](NOTICE). Model weights remain subject to their publishers' licenses.
