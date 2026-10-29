# Typed SSD settings v1

`engine/ggml/src/ggml-ssd/settings/registry.def` is the authoritative settings registry.
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
| `model.optimizations` | strings / exact model rules | `[]` | model | all |
| `model.content_manifest` | strings / ordered shard SHA-256 | `[]` | model | all |

These are conservative parser defaults and retained policy priors, not claimed
per-device performance optima. The framework must provide its measured CPU,
cache and placement configuration. Linux requires at least two total CPU
participants; one is the I/O participant. An explicit compute CPU list must
contain `cpu.threads - 1` unique IDs in declared order. Zero decode participants
leaves the measured/default policy; positive values cannot exceed compute
participants. The VRAM margin is at least reserve plus 64 MiB. Non-applicable
platform keys cannot be explicitly set. On macOS the interface supports static
settings only; it does not include an online adaptive controller.

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
Settings are never translated to environment variables. `inferssd launch --execute`
passes the settings file to the server with `--ssd-settings`, and each setting
with a command-line origin as the matching `--ssd-*` option.

## Exact model optimization rules

`model.optimizations` is a typed `strings` array in the same settings snapshot.
Its registry default is `[]`: no new optimization overrides are enabled. Both
`inferssd configuration write --set` and native `--ssd-model-optimizations`
accept the JSON array literal; launch forwards it through `--ssd-settings` or
its typed CLI override, without environment variables. For example:

```text
inferssd-settings 1
model.optimizations = ["reviewed-example|off|example-uuid|deepseek41|36|2|Example GPU|1|0"]
```

This is a **disabled synthetic example**, not a validated model or recommended
setting. Each string has exactly nine `|`-separated fields:

```text
rule_id|off_or_on|uuid_or_verified_content|general.architecture|general.file_type|general.quantization_version|hardware|engram_prefetch|hc_fused
```

Use the exact scalar values from the loaded GGUF metadata. `hardware` is the
semicolon-joined backend device descriptions in the loaded model's device
order. This scope does not identify the host CPU, driver version or firmware;
a future candidate sensitive to those must extend the binding before enabling it.
Effective values are visible in the native startup `model options` log
and, when telemetry is enabled, the `ssd.model.options` event. An empty device
list or missing identity/quantization metadata cannot match. All identity fields
must match exactly; filenames, launch profile names, model families, substrings
and wildcards do not participate. Models without `general.uuid` can use the
verified content fallback below; otherwise they retain existing settings.
The UUID must have been validated against the exact model
artifact and quantization, including mixed tensor types; metadata is an identity
claim, not an integrity check of the weights.

`off` keeps a rule inactive. `on` is the operator's explicit declaration that the
exact model, quantization, hardware and **final gated source** have been validated;
the runtime does not certify benchmark evidence. Do not enable an unvalidated
entry. Each option is `0`, `1`, or `-` to keep its existing setting. Options
currently gate the existing Engram read-ahead and fused hyper-connection graph
paths in `engine/src/models/deepseek41.cpp`. Matching rule values overlay file
settings; explicit per-option native/launch CLI values win. Existing direct
`model.engram_prefetch` and `model.hc_fused` settings and the launcher's existing
`dsv` defaults retain their behavior when rules are absent, disabled or unmatched.

Resolution happens in each native `llama_context`, before graph construction,
from its owned immutable snapshot and loaded model facts. It changes only that
context's graph options. Consecutive and simultaneous contexts do not share a
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

### Missing UUID: native verified content admission

Actual inspected metadata receipts have no `general.uuid`: DeepSeek reports
`general.architecture=deepseek41`, `general.name=Src`, `general.file_type=38`,
`general.quantization_version=2`; Muse reports `muse-spark`, `Muse Spark Open`,
`14`, `2`. Names are not usable artifact identities. These receipts establish
metadata shape only; they do not establish content hashes or performance profiles.

`model.content_manifest` is a typed string array, default `[]`, containing one
lowercase 64-digit SHA-256 per complete GGUF shard, in the native loader's order.
Use hashes from a trusted artifact receipt, then enable verification at launch:

```text
model.content_manifest = ["<sha256-of-complete-first-shard>", "<sha256-of-complete-second-shard>"]
```

The placeholders above are deliberately invalid. The manifest has no paths and
cannot select another model. It passes through the production launcher's owned
settings file or `--ssd-model-content-manifest` CLI override, with existing
precedence. Native common code passes the same typed options into model loading
and context creation. No environment identity, global flag, extra credential or
authentication mechanism is involved. This is native content verification, not
an unchecked identity asserted by the Python launcher.

When explicitly requested, before tensor loading the native loader hashes the
actual open shard descriptors with the existing vendored public-domain SHA-256
implementation. It checks every shard against the manifest, including all tensor
bytes (and thus mixed quantizations). The fallback identity is exactly:

```python
"sha256:" + sha256("".join(shard_sha256 + "\n" for shard_sha256 in ordered_manifest).encode("ascii")).hexdigest()
```

Use that value in the rule's third field only when UUID is absent. Architecture,
file type, quantization version, hardware, opt-in state and option CLI precedence
still match independently. A UUID-bearing model continues using its UUID.
There are no enabled or validated latest-main profiles shipped here.

The verified manifest, descriptor receipts and shard handles belong to the loaded
model. Each context must supply the same manifest; another model or a stale
claim cannot inherit it. Descriptor device/inode, size, modification and change
timestamps are checked around hashing and again at context creation. Handles
remain open until model destruction. External expert indices must resolve every
expert to the same admitted shard inode and exact byte range; an unrelated,
reordered or packed source is not assumed equivalent. This reuses the native
source-view resolver rather than duplicating source-index parsing/admission.
No applicable source view, an unbound external source, malformed/absent/wrong
manifest, unsupported direct I/O, or Windows descriptor validation leaves content
identity unknown and preserves existing options. In particular, the current Mac
external cache has no source-view binding and cannot yet use this fallback.
I/O exceptions fail loading; they never become admission success. Existing UUID
behavior and direct option settings are retained.

Treat loaded weights and expert sources as immutable for the model lifetime.
This admission guards accidental stale configuration and file replacement; it
is not a defense against an adversary mutating live mapped files after admission.
The startup log reports `content validation startup_us` and the admitted loader
identity; context telemetry reports the usable content identity and matching rule.
Hashing costs one sequential full read of all shards, bounded to a 1 MiB buffer,
plus small per-shard receipts and expert range checks. It can warm the page cache
and adds to model load time. Report it separately from inference throughput and
latency. A/B must use the same manifest, validation and cache conditions; do not
compare a hash-warmed candidate against an unvalidated baseline. Each model load
validates again; there is no persistent verification cache.

Verification uses tiny synthetic two-shard GGUF fixtures with the actual metadata
shapes, independent Python SHA-256 receipts, wrong content/shard lists, absent and
stale claims, quantization mismatches, owned typed model/context plumbing, and
sequential/concurrent admission isolation. These are controller/runtime regression
tests, not model benchmark results. No hundreds-of-GB model was scanned for this
change. THP remains unintegrated: it requires a separate context-safe consumer gate
and new measurements of the final code. Repeated execution remains unimplemented.
