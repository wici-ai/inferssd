<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Translated from README.md at 8b1b6500. The English README is authoritative. -->
<h1 align="center">用你的固态硬盘运行 MoE 大模型</h1>

<p align="center">
  <a href="https://wici.ai">
    <img src="assets/wicissdinfer.png" width="600" alt="InferSSD">
  </a>
</p>

<p align="center">
  <a href="https://wici.ai/wici-infer#app-download"><b>下载应用</b></a> &nbsp;·&nbsp;
  <a href="https://wici.ai/inferssd"><b>官网</b></a> &nbsp;·&nbsp;
  <a href="https://wici.ai/research"><b>博客</b></a> &nbsp;·&nbsp;
  <a href="https://discord.gg/wici"><b>Discord</b></a> &nbsp;·&nbsp;
  <a href="https://join.slack.com/t/wici-ai/shared_invite/zt-4f2e7ob2s-9nUg35Phqyqt4EI6qUjw5w"><b>Slack</b></a> &nbsp;·&nbsp;
  <a href="../LICENSE">Apache-2.0</a> &nbsp;·&nbsp;
  <a href="../README.md">English</a>
</p>
<p align="center">macOS（Metal） &nbsp;·&nbsp; Windows 10/11（CUDA · Vulkan · ROCm 开发中） &nbsp;·&nbsp; Linux（CUDA · ROCm · Vulkan 开发中）</p>

本文译自英文 README，内容以[英文版](../README.md)为准。

**运行比内存或显存还大的模型。** InferSSD 基于 [llama.cpp](https://github.com/ggml-org/llama.cpp)，从固态硬盘运行大型混合专家（MoE）模型。它使用标准 GGUF 文件，提供浏览器聊天界面和 OpenAI 兼容的 API。它可以运行多种量化版本的 **DeepSeek-V4.1-Flash-1017** 和 **Qwen3.8 Flash Next**，以及 **Muse-Spark-Open**、**Qwen3.6-35B-A3B** 和 **Gemma4 E4B**。

## 模型

| 模型 | 参数量 |
| --- | --- |
| DeepSeek-V4.1-Flash-1017 | 总参数 763B / 激活参数 8–16B |
| Muse-Spark-Open | 总参数 816B / 激活参数 42B |
| Qwen3.8 Flash Next | 总参数 177B / A3B |
| Qwen3.6-35B-A3B | 总参数 35B / 激活参数 3B |
| Gemma4 E4B | 总参数 7.46B / E4B |

测试过的 GGUF 文件都按发布者发布的原样使用。每个模型的量化版本、发布者和文件大小见英文 README 的 [Models](../README.md#models)。

## 性能

InferSSD 与 llama.cpp、Colibri 在各台测试机器上的速度对比见英文 README 的 [Performance benchmarks](../README.md#performance-benchmarks)，测量方法见 [benchmark.md](benchmark.md)。

## 开始使用

**桌面应用。** [WiCi Infer](https://wici.ai/wici-infer) 是基于 InferSSD 的免费应用，支持 macOS（Apple 芯片）和 Windows（x64）。它会安装引擎、下载模型并提供聊天界面，不需要使用终端。[下载 macOS 版](https://appapi.wici.ai/versioning/wici-infer/redirect/macos-latest) · [下载 Windows 版](https://appapi.wici.ai/versioning/wici-infer/redirect/windows-latest)

**命令行。** 在 macOS 或 Linux 上：

```sh
curl -fsSL https://github.com/wici-ai/inferssd/releases/latest/download/install.sh | sh
```

在 Windows 的 PowerShell 中：

```powershell
irm https://github.com/wici-ai/inferssd/releases/latest/download/install.ps1 | iex
```

在 macOS 15 或更高版本（Apple 芯片）以及 Windows 10 或 11（x64）上，安装程序会下载该版本预编译的 [macOS](macos-package.md) 或 [Windows](windows-package.md) 包，不需要开发工具。在 Linux 上，它会针对机器的 CPU 和 GPU 从源码编译该版本，大约需要 2 到 12 分钟，需要 Git、Python 3.9 或更高版本、C/C++ 编译器、带 cgroup v2 的 systemd，以及与 GPU 对应的 CUDA 工具包或 ROCm HIP SDK。安装程序在独立环境中工作，不使用 `sudo`。如果它打印了 PATH 说明，请按说明操作。

然后启动模型：

```sh
# 在终端里聊天
inferssd cli -hf mrfakename/DeepSeek-V4.1-Flash-1017-GGUF

# 或者启动网页界面和 OpenAI 兼容的 API，地址是 http://127.0.0.1:8080
inferssd serve -hf mrfakename/DeepSeek-V4.1-Flash-1017-GGUF
```

`-hf` 可以下载并运行 `mrfakename/DeepSeek-V4.1-Flash-1017-GGUF`（Q8_0）、`unsloth/Muse-Spark-Open-816B-A42B-GGUF`（UD-Q4_K_XL）、`unsloth/Qwen3.6-35B-A3B-GGUF`（UD-Q4_K_XL）、`bartowski/Qwen3.8-Flash-Next-GGUF`（IQ4_XS）和 `google/gemma-4-E4B-it-qat-q4_0-gguf`。英文 README 的 [Models](../README.md#models) 中的其他文件，请把它的所有文件下载到同一个文件夹，再用 `-m` 指定第一个文件：

```sh
hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF --include "UD-IQ1_S/*" --local-dir ~/models/deepseek
inferssd serve -m ~/models/deepseek/UD-IQ1_S/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf
```

使用 `serve` 时，在浏览器中打开 `http://127.0.0.1:8080`，或者让任何 OpenAI 兼容的客户端连接 `http://127.0.0.1:8080/v1`。加上 `--host 0.0.0.0` 可以让局域网中的其他设备访问；API 没有身份验证，任何能访问这个端口的人都可以使用它。

第一次用 `-hf` 运行时，会把模型文件下载到 `~/.local/share/inferssd/models`（Windows 上是 `%LOCALAPPDATA%\inferssd\models`），并测量你的机器。用 `--model-dir /path/on/ssd` 选择硬盘。内存不需要设置，InferSSD 会使用当前可用的内存。按 **Ctrl+C** 停止，聊天时也可以输入 `/exit`。

升级和卸载方法见[快速开始](quickstart.md#one-command-installation)。

## 延伸阅读

- [工作原理](architecture.md)
- [测试方法](benchmark.md)
- [快速开始](quickstart.md)
- [版本和引擎资源](engine-resources.md)

## 参与贡献

欢迎贡献，包括硬件测试结果，以及专家 I/O、缓存和批处理方面的改进。详见[贡献指南](../CONTRIBUTING.md)。

## 致谢和许可证

InferSSD 基于 [llama.cpp 和 ggml](https://github.com/ggml-org/llama.cpp) 构建，感谢它们的贡献者和模型适配社区。

InferSSD 采用 [Apache-2.0](../LICENSE) 许可证。内置的 llama.cpp/ggml 和其他第三方组件保留各自的许可证和声明，见 [NOTICE](../NOTICE)。模型权重仍受发布者的许可证约束。
