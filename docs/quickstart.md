<!-- SPDX-License-Identifier: Apache-2.0 -->
# Quick start

[Back to README](../README.md)

Both platforms use the `main` branch of **wici-ai/inferssd**. Choose **[Apple Silicon](#apple-silicon)** or **[Linux / NVIDIA](#linux--nvidia)**, then [prepare a launch](#prepare-a-launch).

## Before you start

- Install Git, CMake, Ninja, a C/C++ compiler, and Python 3.9 or later. macOS also needs Xcode's command-line tools and Metal compiler; Linux needs a compatible CUDA toolkit and `liburing` development files.
- Obtain a complete local [GGUF checkpoint](../README.md#models). Keep all shards together. The launcher reads local files and does not download weights.
- Historical DeepSeek runs used about **473 GiB of GGUF files + 269 GiB of expert packs**. Muse packs added about **450 GiB**. The current framework can index GGUF experts directly; a separate pack is optional and must match its source checkpoint.
- Choose CPU, cache and process-memory budgets for your machine. The README benchmark settings describe their recorded runs, not universal deployment defaults.

```sh
git clone --branch main https://github.com/wici-ai/inferssd.git
cd inferssd
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install ./framework
```

## Apple Silicon

Build the Metal backend and describe the resulting deployment:

```sh
cmake --preset macos-metal
cmake --build build-macos-metal --target llama-server --parallel 4
python3 scripts/engine-resources.py \
  --build build-macos-metal --output engine-resources.json
```

Use `--platform darwin` when launching. The current Mac interface uses typed settings and static launch; online adaptive control is deferred. See [Apple Silicon deployment](mac-launcher.md) for cache settings and memory accounting.

## Muse on Mac

Use the same Metal build, select `--profile muse`, and point `--model` at the first Muse GGUF shard. The deployment descriptor selects `engine/models/templates/muse-spark.jinja`. Keep the complete shard set and use a separate source index or matching pack for each checkpoint.

## Linux / NVIDIA

Install `liburing-dev` on Debian/Ubuntu, or your distribution's equivalent. The production launcher also requires systemd and cgroup v2 with a usable memory controller. Ensure the CUDA compiler is available to CMake; set `CMAKE_CUDA_ARCHITECTURES` if your toolchain needs an explicit target for your GPU.

```sh
cmake --preset linux-cuda
cmake --build build-linux-cuda --target llama-server --parallel 4
python3 scripts/engine-resources.py \
  --build build-linux-cuda --output engine-resources.json
```

Use `--platform linux` when launching. The [Linux resource prerequisites](testing.md#linux-resource-prerequisites) describe I/O, cgroup and pinned-memory requirements.

## Prepare a launch

The public entry point is `inferssd launch`. It consumes a typed settings file, a deployment descriptor and an explicit runtime policy:

1. Prepare `settings.ssd` using the [configuration guide](configuration.md). Inspect supported settings with `inferssd configuration schema`.
2. Prepare a policy JSON using the complete format in the [owned runtime guide](../framework/README.md#explicit-owned-server-execution). Set the platform, memory limits, host reserve, runtime and cleanup budgets for your deployment.
3. Select a fresh output directory and a free loopback port. Replace every `/path/to/` placeholder below.

For Linux DeepSeek:

```sh
inferssd launch --execute --platform linux \
  --resources engine-resources.json \
  --model /path/to/model-00001-of-00010.gguf --profile dsv \
  --settings /path/to/settings.ssd \
  --runtime-policy /path/to/runtime-policy.json \
  --output-dir /path/to/runs/instance-001 \
  --context 8192 --batch 512 --ubatch 512 --port 18181
```

For macOS, use `--platform darwin`, a Darwin policy and `--port 8080`. For Muse, change `--profile dsv` to `--profile muse` and select its first GGUF shard. Batch sizes are explicit examples; tune them within the available memory budget.

If `expert.source` is omitted, the framework builds an alias index from the GGUF metadata without duplicating weight payloads. An optional v1 pack can be indexed with `inferssd source index MODEL PACK DIR`; see [source preparation](../framework/README.md#prepare-a-typed-launch). Use the [deployment resources guide](engine-resources.md) when moving binaries and templates outside the checkout.

Open **http://127.0.0.1:18181** on Linux or **http://127.0.0.1:8080** on Mac after the server is ready, or [connect an API client](../README.md#use-it-with-your-tools). Stop the owned launcher with **Ctrl+C**.

The earlier environment-based launch scripts are historical interfaces. Their commands remain in the [original readme-test guide](https://github.com/wici-ai/ssd-llama/blob/a755355f82bcc52869dd3c1085db59210ced068b/docs/quickstart.md); use the typed interface above for this repository.
