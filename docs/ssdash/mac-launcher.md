# Apple Silicon GGUF launcher

The Mac launcher reads expert slices directly from the original GGUF files. It
checks that every shard is present and regenerates the small alias index from
GGUF metadata on each launch. It does not create a second expert-pack copy.

Install Apple's command-line tools, CMake, Ninja and Python 3.9 or newer. Install
the local GGUF Python package in your Python environment, then build:

```sh
python3 -m pip install ./gguf-py
bash scripts/build-macos-metal.sh
```

Set `SSD_LLAMA_MODEL` to the first shard of the matching model. Keep all shards
in the same directory. For example:

```sh
export SSD_LLAMA_MODEL=/path/to/DeepSeek-V4.1-Flash-Q8_0-00001-of-00010.gguf
scripts/run-macos-ssd.sh dsv server

export SSD_LLAMA_MODEL=/path/to/Muse-Spark-UD-Q4_K_XL-00001-of-00011.gguf
scripts/run-macos-ssd.sh muse chat
```

The filenames and shard counts above are examples; use the files from your model
distribution. `dsv` and `muse` select the expected expert geometry and the matching
chat template. Muse explicitly uses `models/templates/muse-spark.jinja`; its
embedded GGUF template is not used by this launcher. Extra llama arguments follow
the mode, for example `scripts/run-macos-ssd.sh muse chat -n 64`.

The server binds to `127.0.0.1:8080` with one slot. Run one instance at a time.
`SSD_LLAMA_BUILD_DIR` selects both the build and launcher output directory; the
default is `build-mac` under this checkout. `SSD_LLAMA_PORT` changes the port.
The launcher reports physical RAM, the expert cache budget in MiB, and free index
storage before starting. The model's other buffers and the OS also need RAM;
the expert cache limit is not a total process memory limit.

Existing controls remain available during integration:

| Environment variable | Meaning |
|---|---|
| `SSD_LLAMA_MAC_EXPERT_CACHE_MIB` | Nonnegative expert cache budget in MiB; checked for size overflow |
| `SSD_LLAMA_MAC_EXPERT_CACHE_POLICY` | `lfu` or `lru` |
| `SSD_LLAMA_MAC_EXPERT_READ_AHEAD` | Selected-expert read queue depth, 0–4 |
| `SSD_LLAMA_MAC_EXPERT_READ_ALIGNMENT` | Source-read alignment: 1, 4096 or 16384 bytes |
| `SSD_LLAMA_MAC_EXPERT_PREFILL` | 1 enables direct expert reads for prefill; 0 keeps the reference path |
| `SSD_LLAMA_THREADS` / `SSD_LLAMA_PREFILL_THREADS` | Decode / prefill CPU threads |
| `SSD_LLAMA_CTX` / `SSD_LLAMA_BATCH` / `SSD_LLAMA_UBATCH` | Context capacity / batch / microbatch |

The launcher defaults to a 4096 MiB LFU cache, two outstanding selected-expert
reads, 4096-byte read alignment, and direct prefill reads. Prefill can reuse cache
hits, while its misses use temporary buffers without changing decode cache heat
or evicting cached experts. The staging limit is 64 MiB per operator batch.
Explicit values still override these defaults, including zero read-ahead or the
reference prefill path for comparisons.

Device-specific cache and thread recommendations require the matching measurement
report; they should not be inferred from total RAM alone. A 4096 MiB cache is
4 GiB, not 4 GB. Target-device qualification of this integrated revision is ongoing.

Generated indexes live under `~/Library/Caches/ssdash/mac-alias`, or under
`$XDG_CACHE_HOME/ssdash/mac-alias` when set. The launcher owns the alias selection
and replaces it atomically only after generation succeeds. This avoids stale
absolute paths after moving model files. Set the model through `SSD_LLAMA_MODEL`,
not forwarded model, URL or Hub options. Inherited remote-model settings are
also rejected so the loaded model and alias index cannot select different sources.
An inherited expert-pack setting is rejected;
the old pack launcher remains available for explicit pack compatibility.

Normal launches do not pause other applications, change power settings, purge
the OS cache or run automatic benchmarks. The [Mac measurement tools](../../bench/macos/README.md)
record separate numerical and performance evidence. Model validation of the
integrated revision is required before publishing performance claims.
