# Apple Silicon deployment

Apple Silicon uses the same installed `inferssd launch` entry point and typed
settings contract as Linux. `inferssd launch` applies static settings on both
platforms; the online controller is exposed only on Linux through the Python
API and is not implemented on macOS.

```sh
cmake --preset macos-metal
cmake --build build-macos-metal --target llama-server --parallel 4
python3 -m pip install ./framework
python3 scripts/engine-resources.py --build build-macos-metal --output engine-resources.json
inferssd launch --help
```

Use `--platform darwin`, `--resources engine-resources.json`, `--profile dsv` or
`--profile muse`, and `--model` pointing to the first GGUF shard. All shards must
be present. Supply the explicit Darwin runtime policy, settings and fresh output
directory described in the [typed runtime guide](../framework/README.md).

The framework checks GGUF metadata and creates an alias index without duplicating
model weights. It selects the matching chat template from the deployment
resources. `host.expert_cache_mib`, `cpu.threads`, and `--batch-threads` control the
supported cache and thread settings; the native process retains its RSS guard,
cache pin ownership and drain behavior. Model-specific cache sizes need measured
memory evidence and are not inferred from machine RAM alone.

Earlier environment-driven launch scripts are not supported interfaces. Copies
kept under `framework/tests/fixtures` are test inputs only; do not run them as
production commands. See [deployment resources](engine-resources.md) for a wheel
installation that works without a checkout and
[measurement tools](../bench/README.md) for separate numerical and
performance evidence.
