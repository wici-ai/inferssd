# Device performance acceptance

The reference and candidate are built once from the pinned plan on the **same
physical device**. A result from another machine, even one with the same GPU
model, cannot serve as the reference. Both sides need the same model shards,
prompt/template, actual settings, CPU/GPU placement, memory/cache/resource
budgets, toolchain and recorded OS-cache treatment. Direct GGUF and unrestricted H2D remain mandatory. The protected broker
owns source/build/observation evidence; candidate output cannot certify itself.

## Interleaved pairs

Each selected cell (device, model and prompt tier) runs two base/candidate pairs
in ABBA order: base, candidate, candidate, base. Pair 1 is the first base arm
with the first candidate arm; pair 2 is the second candidate arm with the second
base arm. Alternating which side runs first keeps a steady drift, or an effect
of running first, out of the mean. Every arm starts a fresh server and makes one
completion of the fixed 512-token prompt with 256 output tokens. All arms are
kept: nothing is retried and no best arm is chosen. A failed arm makes the cell
`failed`.

For every arm the comparator re-reads the raw SSE and timestamps, exact
request/template IDs, the runtime receipt and cleanup, and checks that the arms
ran in the declared order on one clock, each on its own server. The controller
binds the enclosing artifact to the current run/attempt, reviewed source
identities and reviewed device config.

Five metrics are compared:

- Native prefill = server `prompt_n * 1000 / prompt_ms`.
- TTFT = completion request to first token, separate from native prefill.
- First-half decode = `127 / (t128 - t1)`.
- Second-half decode = `128 / (t256 - t128)`.
- Whole decode = `255 / (t256 - t1)`.

All metrics must be finite and positive.

## Decision

For metric m and pair k, `d_k = ln(candidate / base)` for TTFT and
`d_k = -ln(candidate / base)` for the four throughput metrics, so a positive
value always means the candidate is slower. The cell statistic is the mean
`d̄` over the pairs. Its threshold is

```
T_m = max(ln(1 + floor_fraction_m), k * sigma_log_m / sqrt(pairs))
```

where `sigma_log_m` is the per-pair spread of `d` (its root mean square)
measured by an A/A calibration on the same device, model and tier (see below).
`floor_fraction` (1% for every metric) is set in `thresholds` in
[policy.json](policy.json). Because `sigma_log_m` is itself estimated from a
limited number of calibration pairs, `k` is the one-sided Student-t quantile
whose upper tail equals a normal 3σ tail (P = 0.99865), with the band's pooled
pairs as degrees of freedom. A fixed table in `compare.py` gives the same value
everywhere: 4.277 at 8 pairs, 3.765 at 12, 3.545 at 16, 3.345 at 24, 3.253 at
32, 3.122 at 64 (a count between entries uses the nearest lower entry) and the
normal value 3.0 above 64. Each cell records the `k` it used.

Only whole decode, TTFT and native prefill decide a cell. The first and second
128-token decode rates are sub-windows of whole decode; they are measured,
compared and shown, but cannot make a cell `regressed` or out of band.

| Result | Meaning | Check behavior |
| --- | --- | --- |
| `passed` | Every deciding metric's `d̄` is at or below its threshold, the generated tokens are identical in every arm, a noise band matches the measured conditions and the host check flagged nothing. | Passes. It means no slowdown beyond measured noise, not proof that none exists. |
| `regressed` | Some deciding metric's `d̄` is above its threshold and every pair is slower. | Blocks acceptance. Any other review reasons are listed with it. |
| `needs-review` | A deciding metric's `d̄` is above its threshold but the pairs disagree; a generated token sequence differs (between the sides, or between the two arms of one side); the cell has no noise band, or its band was calibrated under different measured conditions; or the Mac host check flagged an arm. | Blocks acceptance until a person reviews the evidence. |
| `failed` | Missing or invalid evidence, mismatched conditions, incomplete cleanup or a failed arm. | Blocks acceptance; fix the evidence or implementation first. |

Without a matching band no regression can be established, so such a cell is
`needs-review` even when it is clearly slower. Bands come only from 512-tier
calibrations of operator runs, so a 4096-tier cell, and any cell of a job
prepared without the operator bands file (for example with
`stage_selection.py`), has no band and is `needs-review`.

With no real change, a simulation of the A/A case (two pairs per cell, normally
distributed `d`, a band estimated from the stated number of pooled pairs, the
1% floor) gives a false `regressed` rate of about 0.13% per deciding metric per
cell for bands of 8, 16 or 64 pairs, close to the normal 3σ tail of 0.135%, and
about 0.19% just above 64 pairs, where the normal value 3.0 applies. A cell has
three deciding metrics, so its false `regressed` rate is at most about 0.4%.

When cells are combined, the
most severe result wins: `failed`, then `regressed`, then `needs-review`, then
`passed`. A device job succeeds only when all of its cells passed.

Each pair is also kept on its own (`<model>/<tier>/r<k>/comparison.json`) with
its measured values and changes. There, a pair is `needs-review` if any metric
was slower or the tokens differed; this explains the cell but does not decide
it. The cell result is in `<model>/<tier>/comparison.json`.

A `regressed` or `needs-review` cell is never cleared by a later run of the same
candidate. A reviewer decides whether it shows a software regression or an
identified environmental cause; a code fix is a new candidate with a new run.

## Noise bands and calibration

Bands are operator data, kept outside the controller code so that calibrating
needs no controller release. They live in `noise-bands.json` next to the
reviewed device profiles (`inputs.json`) on the preparation host, keyed by
device, model and tier. Each band records `sigma_log` for every metric, the
number of pooled `pairs`, the `conditions_sha256` it was measured under, the
`engine_revision`, the `calibration_jobs` and the `date`. Preparation validates
the file and seals its contents into every device's job configuration, whose
hash the root-staged manifest and the operator envelope already bind; the
envelope also carries `noise_bands_sha256`. The device job decides with exactly
those bands, records their hash, and the hosted replay requires the envelope,
the sealed configuration and the device report to agree before recomputing any
decision. A missing file means no bands, so every timed cell is `needs-review`
until a calibration has been reviewed into the file.

A calibration measures one revision against itself. A maintainer submits a
`/ci calibrate` review, with the same authorization rules as `/ci test`, on a PR
whose head is the revision to measure. Base and candidate are then both that
head, exported and built separately exactly as in a normal comparison, so the
noise includes separate builds. The run selects every enabled device and both
models even though the diff is empty, and runs 4 pairs per cell by default (2 to
4, set on the preparation host) in the same alternation: pair k runs base first
when k is odd. A band needs at least 8 pairs, so it takes two default runs. Instead of a decision, each cell reports its per-pair `d` values,
their mean and root mean square, and the status `calibrated`, or `needs-review`
when tokens differed or the host check flagged an arm. A device job with only
`calibrated` cells succeeds.

[noise_band.py](noise_band.py) turns calibration results into bands. It reads
calibration reports (`aggregate.json`) or retained cell `comparison.json` files
and prints the bands object; `--merge` keeps the cells of an existing file that
the new results do not cover:

```sh
python3 -I bench/ci/noise_band.py --merge noise-bands.json run-1/aggregate.json run-2/aggregate.json > noise-bands.new.json
```

For each cell it pools every input's per-pair `d`. Since base and candidate are
the same revision, the expected `d` is zero, and `sigma_log` is the root mean
square of the pooled values. Results from different revisions or measured
conditions are refused rather than pooled. Flagged cells, cells with fewer
than 8 pooled pairs, device rows without a verified result and report errors
are listed as skipped, never silently dropped. More pooled pairs give a smaller
`k` and a steadier estimate. Review the output before replacing
`noise-bands.json`. As with a
profile or policy change, a changed bands file makes a queued authorization
stale and cancels a running one, so update it between runs; the next prepared
job seals the new bands.

A band applies only while the measured conditions stay the same: the model
files, settings, template, resource limits, OS and loaded libraries all enter
`conditions_sha256`. When one changes, the affected cells report
`needs-review` until they are calibrated again.

## Time budget

From recorded runs, each device job has a fixed part for builds, numerical
checks, short requests and preflights, and one pair of both models adds a
timed part:

| Device | Fixed part | One pair, both models | `/ci test` (2 pairs) | Calibration (4 pairs) |
| --- | ---: | ---: | ---: | ---: |
| cloud-5090 | about 18 min | about 14.7 min | about 47 min | about 77 min |
| dgx-spark | about 15 min | about 22.0 min | about 59 min | about 103 min |
| local-mbp | about 12 min | about 22.3 min | about 57 min | about 101 min |

All stay within the 120-minute device job limit, and each arm stays far below
the 7200-second Mac stage limit.

CI samples have `purpose=ci` and are not formal benchmark samples. The formal
benchmark ([FINAL.md](../FINAL.md)) runs each prescribed 512/4K cell once on the
release source, retaining all attempts without choosing a best result.

## Legacy reference adapter

The legacy reference revision (`LEGACY_REVISION` in `reference.py`, also the
fixed baseline of the checkpoint workflow) does not implement the
typed-settings launcher. Its `bench/native_common.py` has no `typed_snapshot`
or `native_launch_policy`, and its `server_argv` does not emit `--ssd-settings`.
The CI driver requires `typed_settings` for both builds, so it cannot compare
that reference directly to a typed-settings candidate.

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
that mapping and the reference adapter are installed and verified, a comparison
against the legacy reference must remain blocked. PRs whose base already uses
typed settings take the equal-policy path without this adaptation.

### Reviewed binding contract

The protected runner config may carry `comparison_bindings` with exactly `tiny`,
`deepseek` and `muse`. Each model's binding is a JSON object with:

- `schema_version: 1`, `kind: "legacy-to-typed"`.
- `semantic_settings`: all keys of the reviewed 14-key registry generation, including inapplicable platform
  defaults. Values must have the registry's actual JSON types.
- `sides.base` and `sides.candidate`: full `revision`, `profile_sha256`, complete
  `launch_policy`, `images` and `build` objects. The base must be exactly
  `LEGACY_REVISION`; candidate must match the plan's **merge candidate**, not merely
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
`template_relatives` (a nonempty subset of the three model keys; the Muse checkpoint
needs `muse`). Paths must be safe source/build relative paths.
`build_spec(binding, side, revision)` returns a copy for the trusted
build owner; it never edits source or chooses defaults. Supported explicit server
targets are `llama-server`, `inferssd`, and `inferssd-server`.

The binding must come from the reviewed config digest, never a candidate-provided
artifact. Comparison support alone does not enable physical execution: deployment
still needs real protected observation and the driver/build integration.
