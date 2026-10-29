# Typed SSD settings v1

`engine/ggml/src/ggml-ssd/settings/registry.def` is the authoritative settings registry.
The C++ schema/help and packaged Python schema are generated from that table;
the cross-language contract compares the actual C++ output with the installed
Python resource. There are no environment mirrors or legacy-name aliases.

| Key | Kind / unit | Default literal | Producer | Scope |
|---|---|---|---|---|
| `expert.source` | paths / path | `[]` | plan | all |
| `cpu.io_cpus` | cpus / CPU ID | `empty` | plan | all |
| `cpu.decode_participants` | integer / compute participants | `0` | plan | all |
| `prefetch.depth` | integer / predicted experts | `2` | plan | all |
| `place.costs` | costs / named us and GB/s terms | `empty` | plan | all |
| `prefill.cpu_max_tokens` | integer / tokens | `4294967295` | plan | linux |
| `vram.budget` | budget / bytes | `auto` | plan | linux |
| `vram.max_slots` | integer / slots | `0` | plan | linux |
| `vram.reserve_mib` | integer / MiB | `0` | launch | linux |
| `vram.margin_mib` | integer / MiB | `64` | launch | linux |
| `host.headroom_mib` | integer / MiB | `512` | plan | all |
| `host.budget_mib` | integer / MiB | `0` | plan | darwin, windows |
| `model.engram_prefetch` | integer / boolean | `0` | model | all |
| `model.hc_fused` | integer / boolean | `1` | model | all |

These are conservative parser defaults and retained policy priors, not claimed
per-device performance optima. The framework must provide its measured CPU,
cache and placement configuration. The platforms are `linux`, `darwin` and
`windows`; a scope is `all` or the platforms it lists. Expert compute runs on
ggml's threads (`-t`, `-tb`, `-C`, `-Cb`, `--cpu-strict`; see
[how it works](architecture.md#threads)); `cpu.io_cpus` places the executor's
own I/O, prefetch and watch threads, and macOS, which cannot pin threads, only
validates it. Zero decode participants leaves the measured/default policy; a
positive value counts the leading ggml threads of each decode phase, and a
decode call with fewer threads fails. The VRAM margin is at least reserve
plus 64 MiB. Non-applicable platform keys cannot be explicitly set. The `vram.*`
keys size the CUDA expert tier. `vram.budget = auto` gives the tier all the room
the GPU admits and gives no tier where no CUDA backend runs: on macOS and
Windows, on a CPU-only build, or with only Metal or another GPU backend, which
then runs the dense layers while the CPU sink computes every routed expert. An
explicit nonzero budget without a CUDA backend is an error. On macOS the
interface supports static settings only; it does not include an online adaptive
controller.

The host memory ledger has two terms on every platform. The envelope is a finite
limit on this process's memory less a 512 MiB reserve: on Linux the tightest
cgroup v2 `memory.max`; on Windows a memory limit of the process's job, when the
job sets one, against commit; and on macOS and Windows `host.budget_mib` against
the process's own memory (Windows PrivateUsage, macOS phys_footprint). With
`host.budget_mib = 0` and no job limit, the envelope is fixed at startup as the
process's use plus the memory then available, so its room starts at the
available memory less the reserve. The system term is the memory available now
less the floor: Linux MemAvailable, Windows the smaller of available physical
memory and available commit, macOS free, file-backed, purgeable and reusable
pages. The RAM expert cache admits only under a readable envelope. An allocation
that does not fit, such as a compute or work buffer, a scratch buffer or a
server checkpoint, first takes room back from the expert caches: on unified
memory from the GPU tier, then from the RAM cache's coldest slots.

## File and CLI contract

```text
inferssd-settings 1
cpu.io_cpus = "3"  # source=derived; input=measured CPU placement
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

The precedence is command line > file > registry defaults. `inferssd launch`
fills keys still at their registry default with choices for the machine (see
[machine profile](machine-profile.md#launches)) before it writes the native
file. A file must have
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
inferssd configuration write --platform linux --set prefetch.depth=4 --output settings.ssd
```

Output creation is exclusive, and supplied receipts survive round-trip rendering.
Settings are never translated to environment variables. `inferssd launch --execute`
passes the settings file to the server with `--ssd-settings`, and each setting
with a command-line origin as the matching `--ssd-*` option. CPU thread counts and
placement are server threadpool arguments, not settings: the launcher passes
`--threads`/`--batch-threads` as `-t`/`-tb` and `--cpus` as a strict `-C`/`-Cb`
mask; see the [framework guide](../framework/README.md).
