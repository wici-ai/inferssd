# Device readiness workflow

Deploy `integration-smoke.workflow.yml` as
`.github/workflows/integration-smoke.yml` on protected `main` in the private
controller repository, `wici-ai/inferssd-device-ci`. Export with
`export_controller.py --workflow integration-smoke` to activate only this manual
entry point; the full performance workflow remains a reviewed template under
`control/bench/ci/controller/` until its own prerequisites are configured. This template checks no public PR out, requests no App/secret, and
performs no source download or runner registration.

The device allowlist is exactly `cloud-5090`, `dgx-spark` and `local-mbp`.
Readiness does not activate the full PR performance matrix.

The operator provisions these fixed Linux paths before dispatch:

- `/opt/inferssd-ci-control/current/venv/bin/python`: root-owned native Python
  3.9 or later environment containing the reviewed control dependencies; the installation
  and interpreter target must be protected from the runner and candidate UIDs.
- `/opt/inferssd-ci-control/current/bench/ci/staged_integration.py`: reviewed
  trusted coordinator, with all imported control files root protected. `current`
  must be an actual protected directory, because the path verifier rejects
  symlink ancestors. Paths are fixed deployment configuration, not dispatch inputs.
- `/opt/inferssd-ci-integrations/<job_id>.json`: root-owned reviewed job config,
  readable by the dedicated controller UID, binding the full source `revision`,
  exact root broker manifest hash, build paths/settings and short model request.
- `/run/inferssd-ci-broker/broker.sock`: active reviewed root broker. Its source
  trees, model inputs and fresh per-job manifest are already staged by the
  operator. The existing device lease, STOP and isolated execution remain in force.

On Darwin the corresponding fixed paths are
`/Library/InferSSD-CI/controller/current`,
`/Library/InferSSD-CI/integrations/<job_id>.json`, and
`/Library/InferSSD-CI/control/executor.sock`. The reviewed control Python is at
`<control>/venv/bin/python` on both platforms. No host or executable path is a
dispatch input.

Use a repository-scoped runner registered **only** with the private controller
repository, carrying `self-hosted`, `inferssd-device-ci`, the selected device,
and its native OS/architecture labels: Linux/X64, Linux/ARM64, or macOS/ARM64.
Its process runs as the dedicated controller UID, with no sudo/worker privilege
or access to another repository's workflows. The concurrency group matches the selected device
performance job; the device lock remains the final exclusion.

After reviewing those bytes and identities, dispatch on `main`:

```sh
gh workflow run integration-smoke.yml --repo wici-ai/inferssd-device-ci --ref main \
  -f device=cloud-5090 \
  -f job_id=FULL_32_CHARACTER_JOB_ID \
  -f source_revision=FULL_40_CHARACTER_SOURCE_SHA \
  -f config_sha256=FULL_64_CHARACTER_CONFIG_SHA256
```

The inputs are data passed through environment variables, validated before use.
The controller rechecks config ownership/hash and source revision; candidate code
runs only in broker stages. A fresh output envelope includes the workflow run ID
and attempt, with the job ID as the execution subdirectory. Controller logs and
all available build/native/cleanup evidence are uploaded even if the attempt
fails. A failed job ID is not automatically retried or reused; preserve it and
stage a new reviewed job if a correction requires another attempt.

The config binds `device`, `paths.platform`, a `numerical` command and one or
two distinct `smokes` (DeepSeek/Muse). Build targets include `test-backend-ops`;
its exact executable from the read-only build is included in `image_relatives`.
The numerical stage performs small actual CUDA0/Metal ADD operations against the CPU
reference and rejects an empty/skipped backend set. This is a backend smoke,
not full model-logit parity. Each model gets a new process and one eight-token
completion. Mac can validate both already installed models under the same lease
without rebuilding. All server images must match that one read-only build.

The report separates `functional`, `evidence`, `cleanup` and `performance`.
`evidence=sealed-locally` means the evidence was hashed on the device. It must
still be independently verified after artifact collection; only then can an
operator report uploaded evidence as verified.
`performance=not-measured` never overwrites an older inconclusive comparison.
The raw evidence manifest hashes build, native, numerical and cleanup records.

The report remains `purpose=integration-smoke`, `performance_acceptance=false`,
`benchmark_samples=0`. One successful model smoke demonstrates that the staged
build/start/inference/cleanup chain works. Each device must actually complete
this chain before all three are ready.
This is not a performance comparison, a merge gate, or a formal benchmark sample.
