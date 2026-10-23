# InferSSD

InferSSD is an SSD-backed inference framework for large mixture-of-experts models. It streams expert weights
from local storage and manages execution across the CPU, host memory and GPU. The Linux adaptive framework
measures the machine and derives the runtime configuration.

InferSSD builds on [llama.cpp](https://github.com/ggml-org/llama.cpp) and ggml. This repository includes the
engine source with InferSSD's runtime extensions. The upstream baseline and component attribution are
recorded in [PROVENANCE.md](PROVENANCE.md) and [NOTICE](NOTICE).

The initial source import is under development toward **v0.1.0**. Published performance results and a complete
platform support matrix will follow validation of the release candidate. Historical engineering logs describe
earlier code and are not release benchmarks.

## Models and platforms

The supported targets are DeepSeek-V4.1 Flash with MXFP4 experts in the Q8_0 GGUF distribution, and Muse Spark
Open UD-Q4_K_XL. Linux CUDA development targets RTX 5090, RTX 5060 Ti and GB10. Other inherited models and
backends have not been qualified by this project. Apple Silicon integration includes a
[GGUF launcher for both models](docs/inferssd/mac-launcher.md); target-device qualification is in progress.

The model files are large: allow approximately 505-508 GB for each model, plus working space. Linux reads
expert ranges directly from the GGUF shards and does not require a separate expert pack.

## Linux prerequisites

- A supported NVIDIA GPU visible to `nvidia-smi`, with CUDA 13.x and a matching driver.
- CMake, a C/C++ compiler, Python 3.9 or newer, and Bash 4.3 or newer.
- Linux cgroup v2 and a systemd user instance with the memory controller delegated.
- Local storage supporting `O_DIRECT`, and an enabled `io_uring` interface. Install `liburing-dev` for GPU
  prefill reads. The framework needs `nvcc` and a C compiler at runtime for its probes.
- Exclusive use of the GPU during measurement. Build on the target machine; the build script selects its
  CUDA architecture and CPU capabilities. Compilation and normal execution do not require root.

Docker, WSL and network filesystems have not been qualified. The framework may start several short-lived
servers, read several GB of model data and evict model pages from the page cache while deriving settings.

## Build and run

Clone this repository and download every shard of one supported model after accepting its license:

```sh
git clone https://github.com/wici-ai/inferssd.git
cd inferssd
tools/adaptive/build.sh
mkdir -p ./run-data
ROOT="$PWD/run-data" MODEL_DIR=/data/models tools/adaptive/adaptive.sh serve
```

For Muse, also set `MODEL_CONF=tools/adaptive/models/muse-spark.conf`. Set `MODEL` to the first GGUF shard
when the model directory contains multiple matching distributions. The default HTTP endpoint is
`http://127.0.0.1:8080`. See [the framework guide](tools/adaptive/README.md) for retained step outputs and
configuration receipts. Do not expose the server directly to an untrusted network.

Current launchers and environment variables retain their source names during integration. A typed settings
interface and a Python package are planned for the first release. No binary packages are published yet.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for checks and review requirements, [PROVENANCE.md](PROVENANCE.md)
for source history, and [the import manifest](docs/inferssd/import-manifest.tsv) for the origin of every changed
path. The `upstream` branch contains the baseline snapshot. `git diff upstream main` exposes the downstream
changes after integration.

The current tree retains the imported engine layout. Work toward v0.1.0 will separate the SSD runtime into
`ggml/src/ggml-ssd/`, its CPU and CUDA executors into backend-specific `ssd/` directories, and the adaptive
framework into `tools/inferssd/`. These boundaries are not yet implemented in the release candidate; the
engine and its upstream integration are still maintained together.

Support is best effort and applies to the listed models on validated hardware. The first release will qualify
the first request after server startup; concurrent serving and long-running conversation behavior need
separate validation.

## Models and licenses

InferSSD ships no model weights. Obtain the model files from their publishers and accept their licenses:

- [DeepSeek-V4.1 Flash 1017 GGUF](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-GGUF)
  or [the original Flash GGUF](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF): MIT.
- [Muse Spark Open GGUF](https://huggingface.co/unsloth/Muse-Spark-Open-816B-A42B-GGUF): the Muse Model
  License and Usage Policy apply. Products using that model must display "Built with Muse Spark Open".
  Review the publisher's full terms, including the separate agreement required for covered model-as-a-service use.

The project code is distributed under [MIT](LICENSE), with third-party exceptions and attribution in
[NOTICE](NOTICE) and the retained component licenses. Legacy K3 reference math is separately licensed and
scheduled for removal before v0.1.0. DeepSeek, Muse, Meta, Kimi and llama.cpp are trademarks of their owners.
InferSSD is not affiliated with or endorsed by them.
