<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Translated from README.md at 2e2ecff. The English README is authoritative. -->
<h1 align="center">用你的固态硬盘运行MoE大模型</h1>

<p align="center">
  <a href="https://wici.ai">
    <img src="assets/wicissdinfer.png" width="600" alt="InferSSD">
  </a>
</p>

<p align="center">
  <a href="https://wici.ai/inferssd"><b>官网</b></a> &nbsp;·&nbsp;
  <a href="https://wici.ai/research"><b>博客</b></a> &nbsp;·&nbsp;
  <a href="https://join.slack.com/t/wici-ai/shared_invite/zt-4f2e7ob2s-9nUg35Phqyqt4EI6qUjw5w"><b>Slack</b></a> &nbsp;·&nbsp;
  <a href="../LICENSE">Apache-2.0</a> &nbsp;·&nbsp;
  <a href="../README.md">English</a>
</p>
<p align="center">macOS / Metal &nbsp;·&nbsp; Linux / CUDA &nbsp;·&nbsp; Windows（开发中）</p>

本文译自英文README，内容以[英文版](../README.md)为准。

**运行比内存或显存还大的模型**。InferSSD基于[llama.cpp](https://github.com/ggml-org/llama.cpp)，从固态硬盘运行MoE大模型。它使用标准GGUF文件，提供浏览器聊天界面和OpenAI兼容的API。目前支持**DeepSeek-V4.1-Flash-1017**、**Muse-Spark-Open**和**Qwen3.6-35B-A3B**，之后会支持更多模型。

## 模型

下表是测试过的GGUF文件，由社区从对应模型的发布版本转换而来。DeepSeek的Q8_0保留官方位宽：路由专家仍是原生4位MXFP4，FP8张量转为8位Q8_0。详见[权重格式说明](benchmark.md#checkpoint-formats)。

| 模型 | 参数量 | GGUF来源 |
| --- | --- | --- |
| DeepSeek-V4.1-Flash-1017 | 总参数763B / 激活参数8–16B | [mrfakename · Q8_0](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-GGUF) |
| Muse-Spark-Open | 总参数816B / 激活参数42B | [Unsloth · UD-Q4_K_XL](https://huggingface.co/unsloth/Muse-Spark-Open-816B-A42B-GGUF) |
| Qwen3.6-35B-A3B | 总参数35B / 激活参数3B | [Unsloth · UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) |

## 性能

InferSSD与llama.cpp、Colibri的速度对比见英文README的[Performance benchmarks](../README.md#performance-benchmarks)。测量方法见[benchmark.md](benchmark.md)。

## 开始使用

在Apple Silicon的macOS或带NVIDIA显卡的Linux上安装：

```sh
curl -fsSL https://github.com/wici-ai/inferssd/releases/latest/download/install.sh | sh
```

InferSSD目前从源码安装，引擎会针对你机器的CPU和GPU编译，暂时没有预编译包。编译大约需要2到12分钟，取决于机器。

安装程序在独立环境中工作，不使用`sudo`。它需要Git、Python 3.9或更高版本，以及C/C++编译器。macOS还需要Xcode Command Line Tools；Linux还需要CUDA工具包、`liburing`头文件，以及带cgroup v2的systemd。如果安装时打印了PATH说明，请按说明操作。

然后启动模型：

```sh
# 在终端里聊天
inferssd cli -hf mrfakename/DeepSeek-V4.1-Flash-1017-GGUF

# 或者启动网页界面和OpenAI兼容的API，地址是 http://127.0.0.1:8080
inferssd serve -hf mrfakename/DeepSeek-V4.1-Flash-1017-GGUF
```

Muse用`unsloth/Muse-Spark-Open-816B-A42B-GGUF`，Qwen3.6用`unsloth/Qwen3.6-35B-A3B-GGUF`。使用`serve`时，在浏览器打开`http://127.0.0.1:8080`，或者让任何OpenAI兼容的客户端连接`http://127.0.0.1:8080/v1`。

首次运行会把模型文件下载到`~/.local/share/inferssd/models`（DeepSeek或Muse约470 GiB，Qwen3.6约21 GiB），并测量你的机器。用`--model-dir /path/on/ssd`选择硬盘，或者用`-m /path/to/first-shard.gguf`指定已有的文件。内存不用设置，InferSSD会使用当前可用的内存。按**Ctrl+C**停止，聊天时也可以输入`/exit`。

升级和卸载方法见[快速开始](quickstart.md#one-command-installation)。

## 延伸阅读

- [工作原理](architecture.md)
- [测试方法](benchmark.md)

## 参与贡献

欢迎贡献，包括硬件测试结果，以及专家I/O、缓存和批处理方面的改进。详见[贡献指南](../CONTRIBUTING.md)。

## 致谢和许可证

InferSSD基于[llama.cpp和ggml](https://github.com/ggml-org/llama.cpp)构建，感谢它们的贡献者和模型适配社区。

InferSSD采用[Apache-2.0](../LICENSE)许可证。内置的llama.cpp/ggml和其他第三方组件保留各自的许可证和声明，见[NOTICE](../NOTICE)。模型权重仍受发布方的许可证约束。
