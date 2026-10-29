<!-- SPDX-License-Identifier: Apache-2.0 -->
# Quick start

[Back to README](../README.md)

Both platforms use the `main` branch of `wici-ai/inferssd`. Choose **[Apple Silicon](#apple-silicon)** or **[Linux / NVIDIA](#linux--nvidia)**, then [prepare a launch](#prepare-a-launch).

## Before you start

- Install Git, CMake, Ninja, a C/C++ compiler, and Python 3.9 or later. macOS also needs Xcode's command-line tools and Metal compiler; Linux needs a compatible CUDA toolkit and `liburing` development files.
- Obtain a complete local [GGUF checkpoint](../README.md#models). Keep all shards together. The launcher reads local files and does not download weights.
- Plan storage for the checkpoint. The benchmarked DeepSeek GGUF files take about **473 GiB**; an optional DeepSeek expert pack adds about **269 GiB**, and a Muse pack about **450 GiB**. The framework can index GGUF experts directly, so a pack is optional and must match its source checkpoint.
- Choose CPU, cache and process-memory budgets for your machine. The README benchmark settings describe their recorded runs, not universal deployment defaults.

```sh
git clone --branch main https://github.com/wici-ai/inferssd.git
cd inferssd
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install ./framework
```

## Apple Silicon

Install the command-line developer tools, then CMake and Ninja (for example with Homebrew):

```sh
xcode-select --install
brew install cmake ninja
```

Build the Metal backend and describe the resulting deployment:

```sh
cmake --preset macos-metal
cmake --build build-macos-metal --target llama-server --parallel 4
python3 scripts/engine-resources.py \
  --build build-macos-metal --output engine-resources.json
```

`inferssd launch` applies static typed settings on both platforms; Linux additionally exposes an online controller through the Python API (see the [framework README](../framework/README.md)). Routed experts run on the CPU from SSD, while dense layers use Metal; unified memory does not need to hold the full expert set. See [Apple Silicon deployment](mac-launcher.md) for cache and thread settings.

## Muse on Mac

Use the same Metal build, select `--profile muse`, and point `--model` at the first Muse GGUF shard. The deployment descriptor selects `engine/models/templates/muse-spark.jinja`, which replaces the chat template embedded in the GGUF. Keep the complete shard set and use a separate source index or matching pack for each checkpoint.

## Linux / NVIDIA

Install `liburing-dev` on Debian/Ubuntu, or your distribution's equivalent. The production launcher also requires systemd and cgroup v2 with a usable memory controller. Ensure the CUDA compiler is available to CMake; set `CMAKE_CUDA_ARCHITECTURES` if your toolchain needs an explicit target for your GPU.

```sh
cmake --preset linux-cuda
cmake --build build-linux-cuda --target llama-server --parallel 4
python3 scripts/engine-resources.py \
  --build build-linux-cuda --output engine-resources.json
```

The [Linux resource prerequisites](testing.md#linux-resource-prerequisites) describe I/O, cgroup and pinned-memory requirements.

## Prepare a launch

The public entry point is `inferssd launch`. It takes a settings file, a deployment descriptor and a runtime policy:

1. Write a settings file with every setting at its default, then adjust it using the [configuration guide](configuration.md):

   ```sh
   inferssd configuration write --output settings.ssd
   ```

2. Copy the example runtime policy for your platform, then set `memory_budget_bytes` and `host_reserve_bytes` for your machine:

   ```sh
   cp docs/examples/runtime-policy-linux.json runtime-policy.json   # macOS: runtime-policy-darwin.json
   ```

   The launcher stops the server after `runtime_seconds`. On macOS it also stops the server if memory-compressor or swap use grows beyond the policy's limits. The [owned runtime guide](../framework/README.md#explicit-owned-server-execution) describes every field.

3. Replace every `/path/to/` placeholder below. Each launch writes its logs and receipts to a new directory, `runs/<UTC time>` under the current directory; pass `--output-dir` to choose another new directory.

For DeepSeek:

```sh
inferssd launch --execute \
  --resources engine-resources.json \
  --model /path/to/model-00001-of-00010.gguf --profile dsv \
  --settings settings.ssd \
  --runtime-policy runtime-policy.json \
  --context 8192 --batch 512 --ubatch 512
```

For Muse, change `--profile dsv` to `--profile muse` and select its first GGUF shard. Batch sizes are explicit examples; tune them within the available memory budget. The platform defaults to the current host, and the server listens on `127.0.0.1:8080`; pass `--port` to use another loopback port.

If `expert.source` is omitted, the framework builds an alias index from the GGUF metadata without duplicating weight payloads. For the recorded planning commands, `inferssd source index MODEL PACK DIR` indexes an optional v1 pack; see [source preparation](../framework/README.md#prepare-a-typed-launch). Use the [deployment resources guide](engine-resources.md) when moving binaries and templates outside the checkout.

To create a v1 pack, run the bundled builder on the shards you load. It needs NumPy and PyYAML for the bundled GGUF reader (`python3 -m pip install numpy pyyaml`):

```sh
python3 engine/tools/expert-pack/create-moe-expert-pack.py \
  '/path/to/model-*-of-00010.gguf' -o /path/to/model.expert-pack --align 4096
python3 engine/tools/expert-pack/create-moe-expert-pack.py --inspect -o /path/to/model.expert-pack
```

To launch with the pack, set `expert.source = ["/path/to/model.expert-pack"]` in the settings file. The v1 format records no checkpoint fingerprint, so rebuild the pack whenever the shards change.

Open **http://127.0.0.1:8080** after the server is ready, or [connect an API client](../README.md#use-it-with-your-tools). Stop the owned launcher with **Ctrl+C**.
