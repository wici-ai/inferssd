# Changelog

## Unreleased

First public version of InferSSD.

- Run DeepSeek-V4.1-Flash and Muse-Spark-Open with routed experts streamed from SSD through bounded RAM and
  VRAM caches.
- Support macOS on Apple Silicon with Metal, and Linux with NVIDIA CUDA.
- Configure deployments with typed `inferssd-settings` files and the `inferssd` Python framework package.
- Serve models through the bundled OpenAI-compatible server with browser chat.
- License InferSSD under Apache-2.0; bundled llama.cpp/ggml and other third-party components keep their original
  licenses and notices.
