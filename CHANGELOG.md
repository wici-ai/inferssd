# Changelog

## 0.1.0 - 2026-10-30

First public release of InferSSD, which runs mixture-of-experts models larger than your RAM or VRAM from SSD. InferSSD builds on llama.cpp, reads GGUF weights directly, and serves browser chat and an OpenAI-compatible API.

### Highlights

- **Models:** DeepSeek-V4.1-Flash-1017 (Q8_0) and Muse-Spark-Open (UD-Q4_K_XL), with routed experts streamed from SSD through bounded RAM and VRAM caches. See the [benchmarks](https://github.com/wici-ai/inferssd/blob/v0.1.0/README.md#performance-benchmarks).
- **Platforms:** macOS on Apple Silicon with Metal, and Linux with NVIDIA CUDA on x86_64 and ARM64. Windows support is in progress.
- **One-command install and launch:** the installer sets up a private environment, and `inferssd cli` and `inferssd serve` download a tested model and start it.
- **Automatic setup:** the first launch measures the machine, and the engine sizes its caches from the memory available. There is no memory setting to choose.

### Install

```sh
curl -fsSL https://github.com/wici-ai/inferssd/releases/download/v0.1.0/install.sh | sh
```

This release installs from source, so the engine is compiled for your machine's CPU and GPU. Prebuilt packages are not available yet. Building takes about 2 to 12 minutes, depending on the machine.

The installer needs Git, Python 3.9 or later, and a C/C++ compiler. macOS also needs the Xcode Command Line Tools; Linux also needs the CUDA toolkit, `liburing` headers, and systemd with cgroup v2.

Then start a model. The first run downloads about 470 GiB of model files.

```sh
# Chat in your terminal
inferssd cli -hf mrfakename/DeepSeek-V4.1-Flash-1017-GGUF

# Or start the web UI and an OpenAI-compatible API at http://127.0.0.1:8080
inferssd serve -hf unsloth/Muse-Spark-Open-816B-A42B-GGUF
```

### Known limitations

- `-hf` accepts only the two tested repositories above. Use `-m` for local GGUF files.
- If the CUDA runtime is outside the system library path, the first launch cannot measure the GPU, and prefill runs on the CPU.

### License

InferSSD is licensed under Apache-2.0. Bundled llama.cpp/ggml and other third-party components keep their original licenses and notices.
