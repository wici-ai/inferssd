# Resume a completed checkpoint base

`resume_base.py` handles one narrow recovery for the
[checkpoint workflow](controller/CHECKPOINT.md): the legacy base arm completed
its single 512/256 request, but the checkpoint failed because a verified
controller stop was classified as an abnormal server exit. Recovery revalidates
the retained base evidence and runs only the candidate in a new lease. It does
not retry or select a performance sample. The original checkpoint and benchmark
JSON remain failed and byte-identical.

The root-owned checkpoint configuration may add:

```json
"resumed_base": {
  "artifact_root": "/opt/inferssd-ci-evidence/ORIGINAL_ATTEMPT",
  "manifest_path": "/opt/inferssd-ci-evidence/ORIGINAL_ATTEMPT.manifest.json",
  "manifest_sha256": "FULL_64_CHARACTER_MANIFEST_SHA256"
}
```

The root-owned evidence manifest binds every original artifact file and its
original job, Actions run and attempt. Both arm specifications, source revisions,
model, hardware, resources, cache treatment, reference profile, binding and
comparison policy must equal the original configuration. Only the new
root-manifest path/hash and the candidate settings host path may change; the
settings runtime path and bytes must remain identical. Keep the original base
build and root job manifest at their existing read-only paths.

Before opening the new lease, the controller verifies the unique raw SSE result,
original stage and job cleanup, before/after process identities, current model
identities, the retained base binary, libraries and toolchain, and the original
native runtime receipt. It copies the failed base artifacts unchanged into the
new output and writes separate derived receipts. It then executes only the
candidate alias/build/CUDA validation/512–256 measurement, once.

`checkpoint.json` records `recovered_source`, the original failed run, and the
current lease separately. These are two leases, not one uninterrupted paired
run. The result remains a CI diagnostic comparison with the single-sample
limitations and unchanged slowdown policy, and adds no formal benchmark samples.
