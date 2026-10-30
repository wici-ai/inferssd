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
the GPU admits; on a GPU that shares the host's memory, such as GB10, that is
the host memory envelope below less the slots the next-layer prefetch lands in,
which are all the RAM cache keeps beside a tier there (with `vram.budget = 0`
the RAM cache takes the envelope). Auto gives no tier where no CUDA backend
runs: on macOS and Windows, on a CPU-only build, or with only Metal or another
GPU backend, which then runs the dense layers while the CPU sink computes every
routed expert. An
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
pages. The RAM expert cache admits only under a readable envelope. On a GPU
that shares the host's memory (unified memory, such as GB10), a CUDA allocation
is not charged to the cgroup, so the Linux envelope counts the CUDA expert
tier's storage as used beside the cgroup's own usage: the tier and the RAM
cache share one envelope, and the tier also stays within MemAvailable less the
floor. An allocation that does not fit, such as a compute or work buffer, a
scratch buffer or a server checkpoint, first takes room back from the expert
caches: on unified memory from the GPU tier, whose blocks also answer any CUDA
allocation that would fail while no layer holds them, then from the RAM
cache's coldest slots.

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


## Exact model optimization rules

`model.optimizations` is a typed `strings` array in the same settings snapshot.
Its registry default is `[]`: no new optimization overrides are enabled. Both
`inferssd configuration write --set` and native `--ssd-model-optimizations`
accept the JSON array literal; launch forwards it through `--ssd-settings` or
its typed CLI override, without environment variables. For example:

```text
inferssd-settings 1
model.optimizations = ["reviewed-example|off|name:Example Model|deepseek41|36|2|Example GPU|1|0"]
```

This is a **disabled synthetic example**, not a validated model or recommended
setting. Each string has exactly nine `|`-separated fields:

```text
rule_id|off_or_on|selector|general.architecture|general.file_type|general.quantization_version|hardware|engram_prefetch|hc_fused
```

Use the exact scalar values from the loaded GGUF metadata. `hardware` is the
semicolon-joined backend device descriptions in the loaded model's device
order. This scope does not identify the host CPU, driver version or firmware;
a future candidate sensitive to those must extend the binding before enabling it.
Effective values are visible in the native startup `model options` log
and, when telemetry is enabled, the `ssd.model.options` event. An empty device
list or missing identity/quantization metadata cannot match. All identity fields
must match exactly; filenames, launch profile names, model families, substrings
and wildcards do not participate. Use `name:<general.name>` for an exact loaded
model-name match, including models without a UUID, or `uuid:<general.uuid>` for
an exact UUID match. Bare UUIDs remain compatible with the initial rule format.
Missing or different selected metadata keeps the original settings. Name rules
also work when a model has a UUID; the selector explicitly chooses the field.

These are metadata selectors, not cryptographic identities. Names can be reused
between revisions, and the GGUF file type does not prove every tensor's format.
Only enable a rule for the tested deployment; model artifact hashes, exact
quantization and benchmark evidence belong to experiment preparation and records.
Full payload checks should run separately after download or before a campaign,
with the same cache treatment for A and B, never as part of option selection.

Runtime selection performs no weight hashing, extra shard reads or per-token
identity checks. The unmerged `model.content_manifest` setting and `sha256:`
selector have been removed; old configurations using them are rejected rather
than silently claiming that payload integrity was checked. Existing model loading
and expert-source behavior is unchanged.

`off` keeps a rule inactive. `on` is the operator's explicit declaration that the
exact model, quantization, hardware and **final gated source** have been validated;
the runtime does not certify benchmark evidence. Do not enable an unvalidated
entry. Each option is `0`, `1`, or `-` to keep its existing setting. Options
currently gate the existing Engram read-ahead and fused hyper-connection graph
paths in the existing model consumers, including DeepSeek-V4.1 and Qwen3.8-Flash-Next.
Matching rule values overlay file
settings; explicit per-option native/launch CLI values win. Existing direct
`model.engram_prefetch` and `model.hc_fused` settings and the launcher's existing
`dsv` defaults retain their behavior when rules are absent, disabled or unmatched.

Resolution happens once in each native `llama_context`, before graph construction,
from its owned immutable snapshot and loaded model facts. It changes only that
context's graph options. An empty rule list takes the original direct-option path,
without constructing model identity, serializing options, or emitting a model-rule
receipt. Configured rules emit one startup receipt, not per-token logs. Consecutive and simultaneous contexts do not share a
mutable feature flag. The startup receipt includes all match fields, the matched
rule ID (empty on fallback), and both effective options. Malformed rules,
duplicate IDs and multiple enabled matching rules reject context creation rather
than choosing an arbitrary winner. There are at most 64 rules, each at most 4096
bytes; empty fields, `*`, newlines and invalid option/state values are rejected.
The configuration parser checks the typed array; semantic rule validation occurs
at native context creation before graph execution.

No production rule is shipped or hardware validated by this infrastructure
change, and no speedup is claimed. CPU build, native resolver/settings/context
ownership tests, cross-language schema/round-trip tests and production launch
materialization tests use synthetic fixtures, not model benchmarks. No allocator
or THP policy is changed. A future process-global allocator candidate needs a
separate context-specific opt-in allocation integration, fallback and tests;
adding a rule alone cannot make a process-global operation model-local. Future
features must add real consumer gates and measure the final gated code, with
correctness and exact source/model/settings/hardware evidence recorded separately.

### Integration with architecture-based launch profiles

The launcher continues to select its profile from the loaded GGUF architecture,
with an explicit user profile taking precedence. Exact model rules resolve later
in the native context and do not change profile, template or sampling selection.
Explicit per-option CLI settings continue to win over a matched rule. The
benchmark reference binding includes all 15 current typed settings, including
`model.optimizations`; omitted or drifting rules are rejected. Removed legacy
CPU/cache settings are not reintroduced by this reconciliation.
