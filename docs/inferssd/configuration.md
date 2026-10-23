# Typed SSD settings v1

`ggml/src/ggml-ssd/settings/registry.def` is the authoritative settings registry.
The C++ schema/help and packaged Python schema are generated from that table;
the cross-language contract compares the actual C++ output with the installed
Python resource. There are no environment mirrors or legacy-name aliases.

| Key | Kind / unit | Default literal | Producer | Scope |
|---|---|---|---|---|
| `expert.source` | paths / path | `[]` | plan | all |
| `cpu.threads` | integer / participants | `2` | plan | all |
| `cpu.compute_cpus` | cpus / CPU ID | `empty` | plan | linux |
| `cpu.io_cpus` | cpus / CPU ID | `empty` | plan | linux |
| `cpu.decode_participants` | integer / compute participants | `0` | plan | linux |
| `prefetch.depth` | integer / predicted experts | `2` | plan | linux |
| `place.costs` | costs / named us and GB/s terms | `empty` | plan | linux |
| `prefill.cpu_max_tokens` | integer / tokens | `4294967295` | plan | linux |
| `vram.budget` | budget / bytes | `auto` | plan | linux |
| `vram.max_slots` | integer / slots | `0` | plan | linux |
| `vram.reserve_mib` | integer / MiB | `0` | launch | linux |
| `vram.margin_mib` | integer / MiB | `64` | launch | linux |
| `host.headroom_mib` | integer / MiB | `512` | plan | linux |
| `host.expert_cache_mib` | integer / MiB | `4096` | plan | darwin |
| `model.engram_prefetch` | integer / boolean | `0` | model | all |
| `model.hc_fused` | integer / boolean | `1` | model | all |

These are conservative parser defaults and retained policy priors, not claimed
per-device performance optima. The framework must provide its measured CPU,
cache and placement configuration. Linux requires at least two total CPU
participants; one is the I/O participant. An explicit compute CPU list must
contain `cpu.threads - 1` unique IDs in declared order. Zero decode participants
leaves the measured/default policy; positive values cannot exceed compute
participants. The VRAM margin is at least reserve plus 64 MiB. Non-applicable
platform keys cannot be explicitly set. Mac M1 is a migration stage, not a chip
generation; this interface does not implement the deferred Mac M2 controller.

## File and CLI contract

```text
inferssd-settings 1
cpu.threads = 4  # source=derived; input=measured CPU placement
cpu.compute_cpus = "2-4"
cpu.decode_participants = 2
expert.source = ["/path/to/index # 1.tsv", "/path/to/other.tsv"]
place.costs = "cpu_fixed_us=204,link_gbps=12.3"
vram.budget = auto
```

Integers are unsigned decimal literals, not quoted strings. Source lists are
JSON arrays of nonempty UTF-8 strings; `#` inside a quoted path is preserved.
CPU lists accept ordered comma-separated IDs and inclusive ranges. Cost tables
accept the eleven schema-listed `name=value` terms separated by commas or
whitespace; values must be finite, nonnegative and at most 1e6, with positive
`link_gbps`. Omitted cost terms retain the executor priors until measured.

The precedence is command line > file > registry defaults. A file must have
exactly the v1 header. Duplicate file keys, unknown keys, wrong types, bad ranges
and dependency errors fail with key/source context; later CLI occurrences win.
Trailing file comments are retained exactly after the `#` byte. CLI overrides
receive `origin=cli`; file and default values retain their own location/receipt.
Each context owns a copy of the immutable snapshot; no mutable global or ambient
environment is consulted during parsing.

The C++ `ssd::settings` API returns `[[nodiscard]] result<T>` and status/error
values. No parser, loader or telemetry error is thrown across its public boundary.
Typed accessors return const pointers, including null for an absent/wrong kind,
and never perform an unchecked `std::get`. `load` reads a file; `parse` accepts
explicit content, source name, overrides and platform. `schema_json`, `help`,
`cli_option` and `cli_override` use the same registry. The startup retired-prefix
guard accepts an explicit list of environment names and rejects the declared old
prefixes plus `TASK[0-9]+_`; it accepts `TASKRC` and `TASKDATA`.

The Python package exposes independent usable preparation commands:

```sh
inferssd configuration schema
inferssd configuration check --platform linux --input settings.ssd
inferssd configuration write --platform linux --set cpu.threads=4 --output settings.ssd
```

Output creation is exclusive, and supplied receipts survive round-trip rendering.
This module does not translate new settings to legacy environment variables.
The existing `launch --execute` remains on its admitted runtime contract until
the C++ production consumers and context binding are connected in the same
integration. This foundation batch alone does not claim all old readers retired.
