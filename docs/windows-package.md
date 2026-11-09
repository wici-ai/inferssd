# Precompiled Windows engine package

`Package Windows` builds the engine on GitHub's Windows runner. Users install the resulting ZIP;
their computers do not need Python, Git, Visual Studio, CMake, Ninja or the Vulkan SDK.
The desktop application consumes the same package through `scripts/install.ps1`.

The package targets Windows 11 x64 (build22000 or later) and AVX2-capable CPUs. It includes a
Vulkan backend and loader, with CPU fallback when no usable GPU driver is present. A GPU vendor's
normal graphics driver is needed for GPU execution; the SDK is a CI build dependency only.
This package does not include CUDA. It uses the existing Vulkan path for compatible NVIDIA,
AMD and Intel GPUs.

The archive has one directory, `inferssd-<date.build.commit>-windows-x64`, containing:

- `bin/inferssd.exe`: a relocatable native console entry point, forwarding arguments, standard
  handles and exit status to the package's own Python.
- `bin/inferssd-server.exe`, CPU/SSD probes, engine DLLs, Vulkan loader and Visual C++ runtimes.
- `python/`: pinned python-build-standalone CPython3.12 with the framework and hash-pinned Hub
  client dependencies. No package installation happens on the user's machine.
- `engine-resources.json`: paths relative to the package, including templates and probes.
- `templates/` and `licenses/`.

Beside the ZIP is an `inferssd-package 1` JSON listing with its size, SHA-256, Windows/CPU minimum,
and the same build identity fields as the macOS package. Published bytes are immutable.

## Build and verify

The workflow builds automatically for packaging-related pull requests, making a
`package-windows-x64` Actions artifact available for testing. To produce a production package,
run `Package Windows` on `main`; check `publish` to upload the verified archive and update
`latest-windows.json` through the same release service as macOS. Publishing requires
`APP_VERSIONING_TOKEN` and is disabled for pull requests and other branches.

For a local build, enter an x64 Visual Studio tools environment, install the Vulkan SDK, and run:

```powershell
scripts/package-windows.ps1 -OutputDirectory dist
scripts/check-package-windows.ps1 -Archive dist/inferssd-<label>-windows-x64.zip
```

The check extracts under a different path containing spaces and removes developer tools from
PATH. It audits PE imports, loads the packaged Python's native dependencies, checks CLI/schema
and CPU execution, exercises Gemma's RAM-admission contract, and starts/stops an owned local
HTTP fixture through the packaged Windows guardian/job runtime. It then installs the ZIP using
the normal installer and verifies the installed command. No model weights are downloaded by
these packaging checks, and they do not claim GPU/model quality or throughput validation.

The main app installer is built separately in `wici-ai/wici-infer`; this repository publishes the
precompiled engine it downloads.
