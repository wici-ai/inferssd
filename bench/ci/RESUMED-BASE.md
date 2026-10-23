# Resume the completed tier-off diagnostic base

This narrow recovery handles the completed base from Actions run `39947760323`,
whose checkpoint failed when a verified controller stop was classified as an
abnormal server exit. It does not retry or select a performance sample. The old
checkpoint and benchmark JSON remain failed and byte-identical.

The root-owned checkpoint configuration may add:

```json
"resumed_base": {
  "artifact_root": "/opt/inferssd-ci-evidence/attempt6-be548cc44ab44699bf7494af56f383b2",
  "manifest_path": "/opt/inferssd-ci-evidence/attempt6-be548cc44ab44699bf7494af56f383b2.manifest.json",
  "manifest_sha256": "bf5a8ac41226e0e0e867ce8f782e81c2d89284ac41d3898fcb9dc50fa5e6456a"
}
```

The sealed manifest binds every original artifact file and its original job,
Actions run and attempt. Both arm specifications, source revisions, model,
hardware, resources, cache treatment, reference profile, binding and comparison
policy must equal the original configuration. Only the new root-manifest
path/hash and the candidate settings host path may change; settings runtime path
and bytes must remain identical. Keep the original frozen base and root job
manifest at their existing read-only paths.

Before opening the new lease, the controller verifies the unique raw SSE result,
original stage and job cleanup, before/after process identities, current model
identities, retained frozen binary/libraries/toolchain and the original native
runtime receipt. It copies the failed base artifacts unchanged into the new
output and writes separate derived receipts. It then executes only candidate
alias/build/CUDA validation/512–256 measurement, once.

`checkpoint.json` records `recovered_source`, the original failed run, and the
current lease separately. These are two leases, not one uninterrupted paired
run. This remains a tier-off CI diagnostic comparison with the single-sample
limitations and unchanged slowdown policy. It adds **zero** rows to the formal
FINAL 16-row benchmark and establishes no result for the original 8 GiB tier.
