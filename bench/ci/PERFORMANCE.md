# Performance acceptance during structural migration

The reference and candidate are built once from the frozen plan on the **same
physical device**. No old development-5090 result can serve as the cloud-5090
reference. Both sides need the same model shards, prompt/template, actual settings,
CPU/GPU placement, memory/cache/resource budgets, toolchain and recorded OS-cache
treatment. Direct GGUF and unrestricted H2D remain mandatory. The protected broker
owns source/build/observation evidence; candidate output cannot certify itself.

Each planned model/tier has one attempt per side. The comparator re-reads the raw
SSE and timestamps, exact request/template IDs, wrapper evidence and cleanup. The
manifest and summary must share a run ID, the complete attempt ledger must match,
and their tiers must equal the plan. The controller binds the enclosing artifact
to the current run/attempt, reviewed source identities and reviewed device config.

Five metrics are gated independently:

- Native prefill = server `prompt_n * 1000 / prompt_ms`.
- TTFT = completion request to first token, separate from native prefill.
- First-half decode = `127 / (t128 - t1)`.
- Second-half decode = `128 / (t256 - t128)`.
- Whole decode = `255 / (t256 - t1)`.

All metrics must be finite and positive. There is **no permitted 5% decode or
10% TTFT regression**. Those earlier provisional limits were not accepted.

| Result | Meaning | Check behavior |
| --- | --- | --- |
| `passed` | All five observations are equal or better. | May satisfy the observed performance gate; it does not prove statistical absence of regression. |
| `inconclusive` | At least one valid observation is slower, even slightly. | Blocks acceptance. `observed_slowdowns` identifies the metrics requiring diagnosis. |
| `failed` | Missing/invalid evidence, mismatched conditions, incomplete cleanup or correctness failure. | Blocks acceptance; fix the evidence/implementation before making a performance claim. |

`regressions` is retained as a compatibility field containing the same metric
names as `observed_slowdowns`; it does **not** mean a reproducible regression was
established by one pair. JSON reports distinguish all three outcomes; both
non-passing outcomes remain non-green in GitHub. A single valid pair cannot prove
variance or significance, and no mean, best attempt or automatic retry is used.

## Bounded follow-up, not repeated matrices

An inconclusive cell opens a separate diagnosis for that device/model and affected
phase. Freeze its hypothesis, budgets and stop rule before running; start with
existing I/O/resource traces and only the shortest request that can distinguish
the cause. At most one additional planned base/candidate pair is the default
diagnostic budget; if still unresolved, report the blocker rather than repeatedly
sampling. A short probe cannot replace a 256-token metric. Preserve every original
and diagnostic attempt, identities, order and observation; never select the best.

This comparator deliberately does not auto-clear an inconclusive run from a
later sample. A separate reviewed diagnostic decision must explicitly say whether
there is a reproducible software regression or an identified environmental cause,
with evidence and limitations. Fixing code creates a new reviewed candidate; it
does not mutate the old report. No complex statistical framework or mandatory
whole-matrix repetition is needed for this initial gate.

CI samples have `purpose=ci` and do not increment the formal 16-row ledger. The
formal benchmark still runs each prescribed 512/4K cell once on final S, retaining
all attempts without choosing a best result.

## Required reference adapter before first migration comparison

The accepted old main `f5e6e26ff26d2014cccef0118048039b88163f54` does not implement
the typed-settings launcher. Its `bench/native_common.py` has no `typed_snapshot`
or `native_launch_policy`, and its `server_argv` does not emit `--ssd-settings`.
The current CI driver requires `typed_settings` for both builds, so it cannot yet
compare that reference directly to the final typed-settings candidate.

The controlled adapter in `reference.py` belongs to the protected controller,
with an exact reference-SHA allowlist. Integration must launch the **unchanged old binary** using its
actual reviewed argv/environment, launch the candidate through the new interface,
and record both original launch policies plus an audited mapping to the same
semantic settings and observed budgets. No reference values may be guessed, and
adding a fabricated `typed_values` field is insufficient. Values whose behavior
or resource meaning cannot be aligned must make that cell incomparable. The
adapter validates an explicit binding; it does not invent or approve a real host's
mapping. The driver/build owner must wire the reviewed binding into both builds.

Without an explicit binding, the comparator requires equal loaded-image basenames.
Directory moves and renaming/splitting libraries can legitimately change these.
Use an exact reviewed base/candidate **library-role mapping** together with each
side's actual loaded hashes, rather than globally ignoring image identity. Until
that mapping and the reference adapter are installed and verified, the migration
comparison must remain blocked. Future PRs against a typed-settings main can use
the existing equal-policy path without this one-time legacy adaptation.

### Reviewed binding contract

The protected runner config may carry `comparison_bindings` with exactly `tiny`,
`deepseek` and `muse`. Each model's binding is a JSON object with:

- `schema_version: 1`, `kind: "legacy-to-typed"`.
- `semantic_settings`: all keys of the explicitly bound 14- or 16-key registry generation, including inapplicable platform
  defaults. Values must have the registry's actual JSON types.
- `sides.base` and `sides.candidate`: full `revision`, `profile_sha256`, complete
  `launch_policy`, `images` and `build` objects. The base must be the exact accepted
  legacy SHA above; candidate must match the plan's **merge candidate**, not merely
  the PR head. Profile hashes use `native_profile_sha256`'s canonical JSON algorithm.
- `legacy_mapping`: every semantic key has its exact `value`, nonempty
  `explanation`, and nonempty `inputs` list. No mapping is inferred automatically.

Each launch policy preserves the complete original `args`, `environment`,
`template_sha256` and `alias_sha256`. Candidate also carries the actual complete
`typed_values`, exactly equal to `semantic_settings`; legacy must not claim typed
telemetry. Every observed wrapper revision, profile digest and launch-policy
value must equal the reviewed side. Canonicalization removes only the explicitly
approved launch representation and image-layout differences. Hardware, model,
template/alias hashes, all remaining effective settings, actual resource budgets,
ownership core and cache treatment remain comparable without relaxation.

Mapping input forms are explicit:

| `kind` | Additional fields | Verification |
| --- | --- | --- |
| `environment` | `name`, `value` | Exact original launch environment entry. |
| `argument` | `index`, `value` | Exact original argv element. |
| `effective-setting` | `name`, `value` | Exact wrapper effective setting; nested native metadata cannot be used here. |
| `source-default` | `revision`, `path`, `sha256` | Operator-reviewed meaning of an exact legacy source file; never presented as runtime telemetry. |

Call `verify_source_defaults(binding, source)` before launching the base. It
checks the fixed checkout revision and compares every cited file against both
the reviewed hash and that revision's actual Git blob. It runs no candidate
program. A nonempty explanation is an audit trail, not automatic proof that the
reviewer's semantic interpretation was correct; an operator must review it.

`images` is the complete actual loaded-library list, with `{path, sha256, role}`
per entry. Paths are absolute and hashes must match actual mapped-library evidence.
No unlisted or missing library is accepted. Stable role sets must match across
sides; a reviewed split can map several libraries to one role. The executable is
bound by the full profile/build identity and is not added to the library list
when the native wrapper excludes it.

`build` has `server_target`, `binary_relative`, `image_relatives`, and
`template_relatives` (exactly the three model keys). Paths must be safe source/build
relative paths. `build_spec(binding, side, revision)` returns a copy for the trusted
build owner; it never edits source or chooses defaults. Supported explicit server
targets are `llama-server`, `inferssd`, and `inferssd-server`.

The binding must come from the reviewed config digest, never a candidate-provided
artifact. Comparison support alone does not enable physical execution: deployment
still needs real protected observation and the driver/build integration.
