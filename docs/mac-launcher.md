# Apple Silicon deployment

Apple Silicon uses the same installed `inferssd launch` entry point and typed
settings contract as Linux. The current support stage is static M1; online
adaptive Mac M2 remains a separate task.

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

Environment-driven launch scripts are retired public interfaces. Their preserved
copies are test oracles only; do not run them as production commands. Historical
Mac reports describe their original settings and do not validate the current
revision. See [deployment resources](engine-resources.md) for a wheel installation
that works without a checkout and [measurement tools](../bench/macos/README.md)
for separate numerical and performance evidence.
