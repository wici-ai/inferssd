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
  prefill reads. CUDA builds need `nvcc`; the typed runtime itself does not compile probe shims.
- Exclusive use of the GPU during measurement. Build on the target machine; the build script selects its
  CUDA architecture and CPU capabilities. Compilation and normal execution do not require root.

Docker, WSL and network filesystems have not been qualified. The framework may start several short-lived
servers, read several GB of model data and evict model pages from the page cache while deriving settings.

## Build and run

Clone this repository and download every shard of one supported model after accepting its license:

```sh
git clone https://github.com/wici-ai/inferssd.git
cd inferssd
cmake --preset linux-cuda
cmake --build build-linux-cuda --target llama-server --parallel 4
python3 -m pip install ./tools/inferssd
inferssd launch --help
```

Use `macos-metal` on Apple Silicon, or `linux-cpu` for core/CPU development.
The [typed launcher guide](tools/inferssd/README.md) describes model paths,
explicit resource policies, leases, settings receipts and launch artifacts.
`inferssd launch --execute` is the production package entry point. It consumes
`inferssd-settings 1`; native callers may use `llama-server --ssd-settings FILE
--ssd-telemetry FILE`. `llama-server --ssd-schema` emits the current registry.
Historical launchers are retained for recorded comparisons. No binary packages
or formal release are published by this source integration.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for checks and review requirements, [PROVENANCE.md](PROVENANCE.md)
for source history, and [the import manifest](docs/inferssd/import-manifest.tsv) for the origin of every changed
path. The baseline snapshot is commit `8884f8314ce8ea96d62cd602b6bbf7fc25758a09`, already included in main.
`git diff 8884f8314ce8ea96d62cd602b6bbf7fc25758a09 main` exposes the downstream changes after integration.

The core and backend executors, runtime assembly, common/server features and
Python framework now follow the [module layout](ARCHITECTURE.md). Run
`python3 scripts/inferssd/check-layering.py` for dependency checks. The
[upstream patch inventory](PATCHES.md) identifies integration hooks and model/op
exceptions. Real-device candidate qualification remains separate from source
organization; formal benchmark samples have not been collected.

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
