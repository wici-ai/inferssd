# SSDash framework package: first extraction batch

This package currently validates device schema 2 and aggregates **existing** probe reports. It does not run hardware probes, bootstrap a model, plan settings, or launch a server. Existing `tools/adaptive` and launchers remain the production entry points while the remaining C3 contracts and engine settings/telemetry are implemented.

```sh
python3 -m pip install tools/ssdash
ssdash probe --from-reports path/to/probe --experts path/to/expert-index > device.json
ssdash validate --device device.json
```

Runtime uses only the standard library and supports Python 3.9+. File loading is separate from `aggregate(Reports, ExpertMetadata)`, which never reads the filesystem, environment, CPU topology, or a clock. Both report loading dependencies can also be injected. Artifact records use frozen dataclasses with tuples of frozen/scalar values, including nested lane fits and observations; serialization returns a separate dictionary.

The legacy device-profile script is frozen from F (`8b12192920b339a4ef31c9006d1ebe20304ce45a`) under `tests/fixtures/oracle`, with source blob and SHA-256 recorded. Twenty-one synthetic observation cases were run against that exact script **before** testing the new core. They cover discrete/UMA data, zero versus missing values, SSD knee ties and rounding, per-file upper median, missing reports, duplicate CPU replacement, busy-lane replacement and shape ordering. The full-path queue-depth regex is preserved, including the legacy behavior where a parent directory named `qd8` takes precedence over a `reads-qd1.json` filename. Malformed JSON and unreadable/invalid UTF-8 read reports are skipped as before; a valid primary report never reads its fallback. Invalid encoding in required inputs or the optional file-list/lane text still fails. Nonempty malformed observations and missing required metrics remain failures. Successful supported-input outputs compare byte for byte, including key ordering and absent knee versus null fields. Missing required memory data and malformed CPU rows retain failure rather than become defaults.

The new typed ingestion boundary deliberately rejects numeric artifacts outside the finite floating-point range, booleans used as numbers, numeric strings, and non-integer expert byte sizes; F copied/emitted some such values or failed later on others. Objects, arrays and scalar types must match their declared fields. Two fixtures preserve those legacy outcomes and assert the new explicit rejection. This is input validation, not an arithmetic-equivalence claim for malformed data. Unknown device schema versions and unknown lane fields are rejected; required schema-2 fields are not defaulted. Standalone artifact validation normalizes measurement fields to floats and is not a promise to preserve arbitrary JSON whitespace or unknown extension fields. Existing pre-F schema-2 artifacts missing fields introduced before F need a separate explicit legacy adapter; this batch does not fabricate their missing observations.

The F framework/launcher source archive and existing launcher fixtures are preserved in the private migration evidence; no launcher policy was translated in this batch. The larger archived historical run remains separate from current-F goldens. C2 setting folding, engine telemetry, planner/state-machine packaging, Python launch resolution, and Mac M2 probing are not completed here.

Development checks:

```sh
python3 -m pytest tools/ssdash/tests
cd tools/ssdash
mypy --strict src/ssdash
ruff check src/ssdash tests
ruff format --check src/ssdash tests
```

Use a mypy release supporting the configured Python 3.9 target (verified with 1.18.2). The preserved oracle is intentionally excluded from formatting/lint rewrites.

## Decode cost leaf

`ssdash.model.decode_cost` adds `fit_response(x, y)` and
`price_decode(rows, LayerResponse, cpu_scale)`. It consumes fixed replay rows and
recorded layer observations; it does not run the tier replay, choose a plan, fit
new device measurements, change settings, or replace `tools/adaptive/sim.py`.
`LayerResponse` is a frozen nested placement view, not a new top-level `insitu`
artifact version. A complete schema-3 artifact adapter remains future work.

The F `lstsq`/`price` oracle is preserved verbatim in `fixtures/oracle/sim_f.py`.
Twenty pricing cases and nine fit cases bind every output float by `float.hex()`.
They cover measured-count and empirical-readiness cutoffs, capped/uncapped
fallbacks, missing/empty/null/zero fields, CPU scaling, empty rows, dependent
columns, the scaled pivot threshold and original column order. The historical
input projection is evaluated with F code to produce a new offline function
oracle; its old device run is not presented as a current-F golden.

No summation is reordered, no pivot exchange or fit normalization is added, and
integer versus float inputs are preserved. `fit_response` retains F's zip
truncation and empty/ragged-input failures. The typed placement adapter requires
finite numeric table/beta/count observations and integer `pf_k`; malformed typed
artifacts fail explicitly. Missing and empty readiness counts retain F's fallback
semantics; a zero readiness share is distinct from an absent share, and an
explicit null share fails only if the uncapped fallback uses it. Zero-priced
outputs still fail instead of becoming a default estimate. No device or full
planner-equivalence claim follows from these checks.

## Routing replay leaf

`ssdash.model.routing_replay` exposes `Tier`, `RamCache`, `Replay` and `simulate`.
It consumes already parsed integer expert IDs in ordered segments, tokens and
layers. It keeps F's mutable state, arithmetic and update order; it does not read
files, inspect a device, validate a top-level trace artifact or select a plan.
The production adaptive modules and launchers still use their existing code.

F's verbatim `sim_f.py` also supplies fifteen frozen state transcripts, generated
before the new implementation. Every step records score/count `float.hex()`
values, the original heap order, residents, clocks, rows, segment bounds and
behind-cache state. They cover zero/negative capacity, admission thresholds,
decay, locked ties, stale heap entries, repeated expert IDs, incremental segments,
skip prefixes, empty segments and behind-query invalidation. Composing these rows
with the extracted decode cost leaf also matches F.

This extraction deliberately adds no normalization or input-range validation.
Tier score bumps deduplicate keys, but repeated resident IDs still count as
repeated hits and RAM clock touches. Replay accepts a previously seen segment
with changed contents when its length is unchanged, just as F does; it only
replays appended segments. `behind_hits` requires `behind=True`; calls without
tracking retain the old failure. Zero half-life still fails. These boundaries
are retained for equivalence, not promoted as new artifact-validation rules.

## Prefill lane cost leaf

`ssdash.model.prefill_lane` provides the original `coupled(a, b, n)` and
`lane_layer(lane, n, U, routes, cov, b_e, ...)` pure functions. The lane input is a
read-only mapping of numeric or null cost observations; no artifact loader or
new validation boundary is introduced. Missing, null, zero, and negative rates
retain F's behavior. The return remains `(time_us, cpu_us, moves)`.

The frozen F `prefill.py` oracle and 31 cases preserve every output and selected
intermediate float as `float.hex()`. They cover banker's rounding, ring clamping,
the near-equal coupled branch, pageable versus locked copies, host-cost fallback,
the two device slots, both existing drive scaling terms, route bytes, and the
fewest moves within one read of the minimum time. Inputs are not normalized,
arithmetic is not reordered, and the old prefill module and production callers
remain unchanged. This extraction does not include pass sizing, thresholds,
workload planning, calibration, or device measurements.

## Prefill workload geometry

`ssdash.model.prefill_workload` adds `cut`, `served`, `offloaded`, and `cold_share`.
`PassSizing` describes only the existing optional `ubatch` and `ubatch_p` keys;
it does not parse, validate, copy, or normalize a complete prefill artifact.

The functions preserve F's explicit/placement/global ubatch fallback order,
truthy zero/null handling, negative-batch remainder behavior, first-turn cold
state, follow-up reply forms and half weights, and offload minimum of 32 tokens.
`cold_share` still traverses its input twice: a one-shot iterator is consumed by
the first sum. Zero or cancelling total weights return zero. These existing
boundaries are recorded for equivalence, not recommended as valid workload data.

32 frozen F cases cover the return values, order, integer values and exact float
bits, including composed workload forms. The original source and oracle remain
unchanged. Coverage estimation, pass timing, break-even search, threshold choice,
feedback, artifact validation and production callers remain outside this leaf.

## Prefill model functions

`ssdash.model.prefill_model` adds `coverage`, `pass_seconds`, `seconds`,
`break_even`, and `threshold`, composed with the extracted lane and workload
functions. Its `TypedDict` definitions describe the existing model keys for
type checking. They do not validate, copy, normalize, or fill in a loaded model.
The original adaptive module and production callers remain unchanged.

48 cases were recorded from the complete frozen F call graph before extraction.
They bind results and selected intermediate values with `float.hex()`, ordered
mapping reads, call order, and the existing exception type and message. Cases
cover coverage interpolation/extrapolation and caps; cold/warm tiers; measured
and fallback graph costs; lane scaling; mixed pass sizes; finite, infinite and
missing break-even results; threshold ties, reply weights and one-shot iterators.
Composed expectations come from F, not from the newly extracted leaves.

Python's float `sum` changes between the verified 3.9 and 3.12 runtimes: one
composed case differs by one ULP in its summed costs, even in the frozen F code.
Both runtime transcripts are preserved without changing the original 3.12
goldens. Tests require the corresponding exact transcript and exact agreement
with F on the same runtime; they do not introduce a numeric tolerance.
Frozen transcript selection is limited to Python 3.9 and 3.12. On other versions
those transcript checks are explicitly skipped; the same-runtime F comparisons
still run, without claiming a recorded transcript for that version.

Arithmetic, lookup order, doubling/bisection bounds and tie breaks are preserved.
Missing, null and zero costs retain their distinct behavior, including failures
when used. No validation of cold-state weights or batch sizes is added. An
iterator consumed while constructing offloaded passes remains consumed when the
threshold code later asks for served forms. Model fitting, memory sizing,
feedback, artifact loading, planner integration and device measurements remain
outside this batch.

## Prefill calibration, memory and feedback

The remaining pure prefill functions are grouped by their inputs and purpose:

- `prefill_calibration`: `union_points`, `rest_fit`, `build`, and `check` consume
  supplied routing, bootstrap observations and device facts to build and check
  the cost model.
- `prefill_memory`: `graph_line` and `graph_pass_max` price recorded graph-buffer
  reserves and the largest pass that fits them.
- `prefill_feedback`: `drive_factor` and `feedback` fit supplied observations,
  using an explicit callback for the placement's threshold and ubatch.

The mapping types describe the existing data without validating or copying it.
`build` preserves borrowed nested observations, placement/tier ordering, unit
conversions, union-point overwrite rules and rounding. Memory admission keeps
the graph-versus-placement identity check, lexicographic reserve selection,
unified-memory accounting and original zero/missing-input behavior. Feedback
keeps callback order, exception timing and the existing `recent=0` slice of all
turns. Singular fits and one-shot iterator behavior are unchanged.

67 cases were frozen from F before implementation, separately on Python 3.9 and
3.12. The two runtime records happen to agree for these cases; both are retained.
They record exact results and selected intermediate values, input hashes,
borrowed-object identities, ordered mapping accesses and callback traces. A
composed case builds the model, prices passes, selects a threshold, applies the
memory bound and checks the lane costs through F's complete call graph. Tests
also compare each new implementation directly with F on the running interpreter.

This completes extraction of the pure functions from the old prefill module.
Artifact loading and frozen schema contracts, engine telemetry, settings
derivation, planner/controller integration and production callers remain separate
work. No device is queried by these modules.


## Complete settings generation

`ssdash.plan.settings` now contains the complete configuration/derivation/receipt
core from the frozen framework. It consumes a `MachineView` supplied by the caller
and returns ordered `Setting` tuples. `configuration` and `derive` compose with
`env` and `render`; field types, order, pending comments, receipts, formatting and
the final newline preserve the original behavior. This still emits the original
environment keys. The fourteen-key engine registry and a new settings file schema
remain separate work.

`ssdash.plan.placement` formats the existing `all`, `split`, `pin` and `@ubatch`
choices. `configuration(..., host_threads=provider)` and placement `env` require
an explicit zero-argument host fallback. It is called only for an otherwise valid
split placement whose recorded `default_threads` is absent or falsey. Explicit
recorded threads, pin/all and unsupported split layouts do not call it. The
package does not inspect `/sys` or query the current host's CPU count. An I/O
adapter can supply that original fallback, or an offline caller can return its
captured value. A provider returning `None` or raising preserves the original
failure path; values are not normalized or eagerly queried.

`settings_types` describes existing input mappings and the Machine replay method.
These are typing views without runtime validation or copies. Replay may retain
cache state; its call order and partial state on failure remain observable.
The `Setting` type retains tuple fields, positional construction, the default
`pending=False`, `_replace` and immutability. The existing versioned device
contract in `ssdash.contracts` is unchanged.

The frozen settings tests cover 97 complete and boundary cases, including discrete
and unified memory, calibrated/probe/missing costs, known/forecast workloads,
pending iterators, checkpoint/headroom receipts, prefetch rank costs and lazy host
fallbacks. They compare the full output and mutation/call/exception observations
with original F on Python 3.9 and 3.12. Five cases have different resident-set
iteration order between those runtimes; both original transcripts are retained,
and their final settings/env/render outputs agree. Tests do not infer the frozen
transcript of another Python version. No production CLI, launcher, artifact
loader or online controller is switched by this extraction.

## Offline planner consumer

The installed package now connects the complete Machine model, replay/decode and
prefill cost models, settings generation, candidate rejection, stable ranking
and calibration to an opt-in command:

```sh
ssdash plan recorded-run --prompts 32,17 --decode 8 --window 1-8 --known \
  --pending '' --host-threads 8 -o chosen-plan.json
```

It reads the existing `insitu.json`, `probe/device.json` and `model.json` in that
order. It does not bootstrap, probe hardware, start a server or change the old
adaptive/online entry points. The output retains the old plan JSON fields,
field order, receipts, environment keys and numerical formatting. It writes the
requested output directly, as the original planner did; it does not silently
add atomic publication. All rejected candidates are printed before failure,
and an all-rejected plan produces no output file.

`--host-threads` supplies only a lazy fallback for a valid split placement with
no truthy recorded `default_threads`. This is a new, explicit offline input
boundary: planning an archived run never guesses its CPU count from the current
host. Recorded defaults, all/pin placements and invalid split placements retain
their previous paths. An omitted fallback errors only if it is actually needed.
No extra numerical or schema validation is introduced by this model migration.

The Python `Machine` accepts an `ArtifactReader` and a host callback. It retains
input references and owns mutable replay/sink caches. It reads a saved plan only
when `pending is None`, after prefill construction and configuration generation;
an explicit empty pending sequence still suppresses that read. `run_plan` takes
explicit clock, output and JSON-writer ports. Tests fix the clock transcript to
compare `plan_seconds` and entire JSON/stdout bytes; real CLI timing is measured.

Whole-file F oracles cover the complete old planner, Machine and reload-to-settings
call graph under the measured Python 3.9 and 3.12 runtimes. F's arithmetic,
iteration/tie order, cache reuse and exception timing remain the reference.
Supplemental tests use real temporary artifacts, the installed CLI and original
F state transitions. The saved historical run is only its actual legacy input,
not evidence for modern UMA/lane configurations or a historical F output.

Decode pricing now has one Mapping-based F implementation, `price_mapping`.
The existing frozen `LayerResponse` validation remains available; `price_decode`
is its thin adapter with unchanged successful numerical results. One error text
is deliberately aligned with F: uncapped explicit null readiness raises the
native `TypeError` at exponentiation instead of the adapter's former custom
message. The exception class and empty-range behavior remain unchanged.

This batch does not migrate the settings CLI's ledger-log parser, the fourteen-key
engine registry, telemetry, live control, launch execution or Mac auto-planning.
