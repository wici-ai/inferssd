# SSDash offline framework

This package reads model headers to create source artifacts, and turns recorded
probe reports and arm logs into calibration artifacts, a plan, legacy launcher settings and recorded conversation reports. The commands are opt-in offline consumers;
`tools/adaptive` and the existing launchers remain the production entry points.
The package also provides an injectable controller model with caller-owned observation
sessions and complete request/bootstrap/conversation orchestration with required effect ports.
It supplies no live engine or HTTP executor. The package runtime uses only
the Python standard library.

## Commands and artifacts

| Command | Recorded input | Output |
| --- | --- | --- |
| `ssdash source` | GGUF headers, optional v1 pack index, dense placement text | Expert JSON/alias TSV, source facts, lane arguments or fio iolog text |
| `ssdash probe` | Existing probe reports and expert metadata | Device schema-2 JSON on stdout |
| `ssdash validate` | Device schema-2 JSON | Validation result; no rewritten artifact |
| `ssdash fit` | Named arm directories, optional graph arm and ubatch JSON | Model schema-2 JSON, insitu schema-3 JSON and a report |
| `ssdash plan` | `insitu.json`, `probe/device.json`, `model.json` | Plan JSON and ranked candidates |
| `ssdash settings` | The recorded artifacts, saved plan and optional ledger logs | Legacy `settings.env` on stdout or at `-o` |
| `ssdash bootstrap-records` | Recorded CPU labels or ubatch run/server logs | CPU-set candidates or legacy `{first, held}` JSON |
| `ssdash report` | Saved plan, conversation events and server logs, optional drive/device facts | Comparison text and legacy `compare-online.json` |

These commands do not run hardware probes, bootstrap a model, start a server or
control an online session. Install from the repository root:

```sh
python3 -m pip install ./tools/ssdash
```

Source artifacts can be generated without reading tensor payloads or running a probe:

```sh
ssdash source index models/model.gguf recorded/expert-index
ssdash source facts models/model.gguf recorded/expert-index/experts.tsv
ssdash source lane-args recorded/expert-index
ssdash source dense models/model.gguf recorded/dense-homes.tsv recorded/dense.tsv
ssdash source iolog recorded/expert-index 100 17 recorded/reads.iolog
```

`source index MODEL [PACK] DIR` optionally uses a v1 expert pack index.
`source iolog DIR N SEED OUT [FILE]` optionally selects one source file; it writes
fio input text and does not invoke fio or read the referenced payload ranges.
`dense-homes.tsv` contains `tensor<TAB>ram|drive` lines. These commands retain the
legacy artifact formats and consumed-input errors. The new CLI uses argparse
for missing/invalid command arguments rather than the old script's usage text.
A generated alias is not a payload-integrity
certificate or a substitute for the Mac launcher's stricter alias validation.

Recorded bootstrap inputs and results can be read separately:

```sh
ssdash bootstrap-records coresets recorded/cpus.txt
ssdash bootstrap-records tier-slots recorded 32 64 128 > recorded/ubatch.json
```

The CPU-row file contains `cpu capacity max_khz [thread_siblings]`. Candidates
keep the recorded sibling filtering, 92% grouping against the first member of a
group, stable ranking and duplicate behavior. `tier-slots` reads each recorded
`ubprobe-UB.out` first line and that run's `server.log.gz`; repeated UB arguments
are read again and keep their original JSON key position. Neither command
queries the current machine. Empty/malformed rows and the legacy zero/None tier
cases retain their original errors; only consumed `OSError`s in `tier-slots`
become null entries. These commands provide recorded inputs for later steps and
do not run native bootstrap or fix the legacy parser's edge cases.

A complete recorded calibration path, assuming the destination directories already exist:

```sh
ssdash probe --from-reports records/probe --experts records/expert-index \
  > recorded/probe/device.json
ssdash validate --device recorded/probe/device.json
ssdash fit --run all=records/all --run pin=records/pin --skip 128 \
  --profile recorded/model.json -o recorded/insitu.json
ssdash plan recorded --prompts 512 --decode 256 --window 1-256 -o recorded/plan.json
ssdash settings recorded -o recorded/settings.env
ssdash report recorded
```

`fit --graph DIR` reads an existing graph arm. `--ubatch-probe FILE` accepts the
legacy numeric or `{first, held}` ubatch JSON. Neither option executes a probe.
The first named placement is the reference; repeated names are all measured and
replace the earlier value without moving its insertion position.

`plan` also accepts `--known` and `--pending`. `settings` supports `--tier`,
`--cpus`, `--prompts`, `--decode` and `--ledgers DIR ...`. Use each command's
`--help` for its arguments. For a valid split placement without truthy recorded
`default_threads`, supply `--host-threads` to `plan`, `settings` or `report`. This explicit
fallback is requested only when needed. The offline commands do not infer a
recorded machine's CPU count from the current host.

## Compatibility and failure behavior

The reference framework F is frozen at
`8b12192920b339a4ef31c9006d1ebe20304ce45a`; verbatim source oracles and synthetic
observations are under `tests/fixtures`. Extraction preserves arithmetic, sum and
mapping order, stable ties, rounding, mutable replay state, borrowed references,
callback order and exception timing. Composed expectations come from the complete
original F call graph, including request output → calibration → plan → reload → settings
→ Controller → recorded report.
Toy logs are synthetic inputs, not claimed engine-run evidence. Historical
archives establish only their actual recorded input coverage.

The installation requires Python 3.9 or later. Normal-wheel regression and exact
F transcripts are verified on Python 3.9 and 3.12. Their original results can
differ because of float summation, resident-set iteration or native exception
wording. Tests retain separate runtime transcripts and compare exactly within
each runtime, without a numeric tolerance or production version branch. Frozen
transcript checks select only those two versions; direct F comparisons that do
not require a recorded transcript still run. Future versions are not assumed
identical to either verified runtime.

Most legacy artifact and model mappings are typing views, not new schemas. They
are not copied, normalized, validated or filled with defaults on ingestion.
Missing, null, empty and zero values retain distinct behavior, including failures
when consumed. For example, replay keeps F's length-only incremental-segment
check and duplicate-hit behavior; workload iterators retain their consumption
order; singular/empty fits are not repaired. `fit_response` returns a tuple rather
than F's list, and calibration immediately consumes it in the original rounded
list comprehension. Complete-command tests bind those results and failures.

Two explicit typed APIs have stricter boundaries:

- `DeviceProfile` validates schema 2 and uses frozen nested records. It rejects
  unknown schema versions and lane fields, missing required fields, numeric
  strings, booleans as
  numbers, non-finite/out-of-range numbers and non-integer expert byte sizes.
  Validation normalizes measurement fields to floats; it does not promise to
  preserve arbitrary JSON whitespace or unknown extension fields. Old artifacts
  missing required observations need an explicit adapter, not fabricated facts.
- `LayerResponse` validates finite layer/count/beta values and integer `pf_k`.
  Its `price_decode` adapter uses the single Mapping-based F pricing core,
  `price_mapping`, and preserves successful results. Uncapped null readiness
  raises F's native `TypeError` at exponentiation rather than the adapter's former
  custom message; the exception class and empty-range behavior are unchanged.

The recorded probe loader preserves the full-path queue-depth regex, even when a
parent directory's `qd8` precedes the filename's `qd1`. Malformed/unreadable read
reports, including invalid UTF-8, are skipped; a valid primary report does not
read its fallback. Invalid encoding in required inputs or file-list/lane text
still fails. Nonempty malformed observations are not silently defaulted.

### File and output ordering

Source header reads retain the original mmap lifetime and decoding: metadata
strings use replacement decoding, while textual artifacts and pack tensor names
use strict decoding. Split-model discovery uses the original sorted glob and does
not newly reject missing shards. Type 42 (`F8_E4M3_B128`) remains recognized as
historical header metadata; this does not claim native execution support. Its
metadata compatibility is separate from the C1 native format retirement work.

`source index` retains mtime-based reuse, including the original set iteration over
source paths. Frozen filesystem transcripts use `PYTHONHASHSEED=0` as a test
condition; dynamic comparisons also check another seed. Production iteration is
unchanged. GGUF alias output is closed and replaced before the expert JSON is
written. Failure can leave a directory, temporary file or the already-published
alias, as in F. Pack indexing does not create a GGUF alias. Dense placement keeps
input order and validates each consumed line before writing its alias. No new
whole-artifact validation, transaction, random seed default or filesystem probe
is inserted into these paths.

Legacy arm parsing prefers a readable gzip server log; corrupt gzip is an error,
not a reason to retry the plain file. Server text uses replacement decoding,
while client JSONL and metadata use strict decoding. A supplied incremental parser
skips the server log but rereads client/metadata. The standalone ledger reader
requires a record at the start of a line; the full parser retains its different
prefix-stripping behavior. Parser construction and package imports do no file,
host or process work.

`fit` directly writes the compact model profile before drive normalization and
optional ubatch processing. It writes the indented insitu artifact before printing
the report. Later errors can leave one or both files, including partial output.
The temporary streams passed to `json.dump` retain F's implicit-close behavior;
there is no transaction or new explicit `with` block.

`plan` reads insitu, device and model artifacts in that order. A saved plan is
read lazily only when `pending is None`; an explicitly empty pending sequence
suppresses that read. All rejected candidates are printed before failure, and
an all-rejected plan creates no output. Successful plan JSON is written directly.
The injected clock is fixed in exact-output tests; real CLI `plan_seconds` uses
the actual elapsed time.

`settings` retains the legacy `settings.env` format and ordered values/receipts,
including its historical `tools/adaptive/settings.py` comment label. The label
identifies the retained format, not the executable used. Writing to a file uses
`OUT.tmp` followed by replacement; write, close or replacement failure can leave
the temporary file, as in F. With no `-o`, output goes to stdout.

`report` keeps the complete legacy comparison: static prefill predictions by route
and tier state, controller decisions, decode-placement flags and final ordered
JSON. It chooses gzip by existence, whereas the arm reader chooses by readability.
It reads the plan before constructing a Machine; only the original OSError,
KeyError and ValueError classes suppress an unavailable model's route checks.
Other errors, including late zero division, retain their point of failure and
already printed text. The report's JSON is written last and can remain partial.
Report text and field names remain legacy formats, not new measurement claims.

## Python interfaces

| Area | Modules and boundary |
| --- | --- |
| Source metadata | `source.header`, `source.experts`, `source.files`, `source.command`: header/parser and legacy artifact operations with explicit filesystem, output and random ports |
| Device facts | `contracts`, `probe.device_profile`: frozen validated device records and pure report aggregation |
| Recorded I/O | `probe.recorded`, `telemetry.files`, `telemetry.recorded`, `telemetry.legacy`: injectable file access and legacy arm observations |
| Cost and replay | `model.decode_cost`, `model.routing_replay`: fitting/pricing and mutable tier/RAM/replay state |
| Prefill | `model.prefill_lane`, `prefill_workload`, `prefill_model`, `prefill_calibration`, `prefill_memory`, `prefill_feedback`: pure costs, geometry, calibration and explicit feedback callbacks |
| Calibration | `model.observed_calibration`, `probe.calibrate`: arm measurements, working set and complete ordered `fit` orchestration |
| Planning | `model.machine`, `plan.planner`, `plan.settings`, `plan.placement`: artifact-backed model, ranked plans and ordered settings |
| Request protocols | `bootstrap.client`, `bootstrap.requests`, `bootstrap.ports`: streamed legacy client timing and cold/warm/graph bootstrap requests with required transport, file, clock and device-number ports |
| Conversation and report | `loop.conversation`, `loop.report`: complete Controller consumer and recorded comparison command |
| Controller model | `loop.records`, `loop.model`, `loop.controller`, `loop.ports`: session-owned observations, re-planning and explicit transition/effect ports |
| Command adapters | `plan.recorded`, `plan.settings_command`, `plan.settings_recorded`: lazy artifact/host/output ports |

`Machine` borrows its input mappings and owns mutable replay/sink caches. Its
artifact reader and host fallback are injected. Command functions take explicit
read, output and clock ports as applicable. Source operations borrow their filesystem ports; import and adapter construction
perform no file or process work. These interfaces preserve existing
semantics; they are not a claim that every legacy artifact now has a versioned,
validated contract.

### Controller model and observation ownership

`ObservationSession(files=...)` owns the arm cache and incremental parser/offsets.
One session preserves one isolated F online module's behavior; independent sessions
are isolated rather than sharing F's hidden globals. Reusing a session explicitly
shares its cache. A gzip key retains directory, skip, size and mtime; a live stream
keeps incomplete trailing lines and the original stale-offset behavior after
truncation. This is the legacy observation format, not versioned engine telemetry.

`Controller` requires `machine_factory`, `engine_factory`, `post`, `effects` and
`session` keyword arguments. The Machine factory can provide the artifact-backed
`model.machine.Machine`; its lazy artifact and host ports still apply. Engine,
HTTP, files, environment, clock and stdout calls occur at their original sites.
No native Engine, HTTP default, process discovery or live CLI is provided.
The old Engine's port-based termination fallback is not copied into the package.

Transitions retain F's mutation and failure behavior: `prepare` returns borrowed
lists, `note` mutates its supplied record, and `after` appends the turn before its
event/count update. A displacement catches only `OSError`. `close()` stops the
current engine without explicitly closing the event stream, as in F. Applications
must not treat this model-only extraction as a complete native resource owner.
The synthetic tests use fake engines and HTTP callbacks; they establish no real
server lifecycle, online performance or device integration claim.

### Request and conversation effects

`turn` requires a transport and nanosecond clock. `run_client`, `run_bootstrap`
and `run_conversation` require every consumed effect port; no network, engine,
host-reader or process implementation is selected implicitly. Their argument
records are unvalidated views of the old parsed arguments. Request URL/body/header
bytes, history aliases, token callback order and sparse first-observed token
timestamps are retained. This legacy timing protocol is separate from
`bench/final_chat.py`, which remains authoritative for formal benchmarking.

Bootstrap preserves cold chat followed by an uncached warm prefill, graph
sentence-cut search, and optional token-count differences. Its drive counters
are whole block-device reads, including unrelated readers; supplied stat/diskstats
ports do not establish per-process I/O. The graph command retains SystemExit(0)
after its output and before drive discovery. The script client opens output
before reading input and records per-turn Exceptions. The conversation consumer
catches only OSError; other failures do not acquire a newly invented finally
cleanup. Its normal path still stops the Controller then records done. These
interfaces are not a native resource-ownership adapter or a live CLI.

The F observations freeze at command return. A fixture may later close its retained
event stream during object collection; that test cleanup cannot mutate a returned
transcript. Tests separately preserve the original open event stream and internal
borrowed-reference behavior.

## Remaining integration boundaries

The old adaptive production entry points have not been switched or deleted.
Live probe/bootstrap execution, Python launch resolution, connecting the controller
to a real online server and process ownership remain separate work. The fourteen-key engine settings registry,
versioned `settings.ssd`, versioned engine JSONL telemetry and removal of temporary
legacy log readers are also pending. This package does not complete C3 by itself.

Mac automatic framework planning and the common CPU executor/cache belong to the
later **Mac M2 migration stage**; “M2” here names a migration stage, not an Apple
chip requirement. Running these offline Python commands on macOS does not imply
that Mac probes, unified-memory budgeting or the shared executor are integrated.
Existing Mac launch/engine paths remain separate from this package.

## Development checks

```sh
python3 -m pytest tools/ssdash/tests
mypy --config-file tools/ssdash/pyproject.toml tools/ssdash/src
ruff check tools/ssdash
ruff format --check tools/ssdash
```

Use a mypy release supporting the configured Python 3.9 target (verified with
1.18.2). CI installs the normal package before running its tests and these checks.
The preserved oracle sources are intentionally excluded from formatting/lint
rewrites.
