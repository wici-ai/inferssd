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
