# Engine deployment resources

The installed framework does not discover a source checkout. Pass `--resources`
to bind one deployment, or provide both `--binary` and `--template`. The optional
`--repo` adapter uses `engine/models/templates/` in a development checkout and
the first `bin/inferssd-server` it finds under the preset build directory
(`build-linux-cuda`, then `build-linux-cpu` on Linux; `build-macos-metal` on
Darwin), then under `build-mac` (Darwin only) or `build`.

A descriptor is ordinary JSON. Relative paths resolve against its own directory:

```json
{
  "schema": "inferssd-engine 1",
  "binary": "bin/inferssd-server",
  "revision": "the-deployed-source-revision",
  "templates": {
    "dsv": "templates/deepseek-ai-DeepSeek-V4.1.jinja",
    "muse": "templates/muse-spark.jinja"
  },
  "probes": {},
  "capabilities": {
    "settings": "inferssd-settings 1",
    "telemetry": 1,
    "platform": "linux"
  }
}
```

Use `darwin` for an Apple Silicon deployment. `--binary` and `--template` override
those individual descriptor paths. Missing selected resources or incompatible
contract versions fail before engine startup. The descriptor is an operator's
explicit deployment contract, not a claim that untrusted binaries were verified.
Normal process ownership, memory policy and runtime evidence checks still apply.

From a source checkout, generate the descriptor after building:

```sh
python3 scripts/engine-resources.py --build build-linux-cuda --output engine-resources.json
```

This inspects the schema without running a model. It describes existing paths;
for a standalone deployment, copy the executable and its runtime libraries plus
templates, then use paths relative to the deployed descriptor. The framework
wheel needs only its standard-library Python runtime. Existing source indexing
uses packaged code and reads GGUF metadata directly, without importing engine
source files or requiring a separate GGUF Python installation.

Use the [typed runtime guide](../framework/README.md#explicit-owned-server-execution)
for the required resource policy and settings. Replacing `--repo` with
`--resources /deployment/engine.json` works from any current directory. The
optional `probes` map names `io` and `memory` executable resources; it does not
trigger automatic probing, and the launcher does not orchestrate a live
bootstrap.
