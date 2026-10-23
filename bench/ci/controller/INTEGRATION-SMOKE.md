# First private integration workflow

Deploy `integration-smoke.workflow.yml` as
`.github/workflows/integration-smoke.yml` on protected `main` in the **private**
`wici-ai/inferssd-device-ci` repository. Export with `export_controller.py --workflow integration-smoke` to
activate only this manual entry point; the full performance workflow remains a
reviewed template under `control/bench/ci/controller/` until its own prerequisites
are configured. This template checks no public PR out, requests no App/secret, and
performs no source download or runner registration.

The operator provisions these fixed paths before dispatch:

- `/opt/inferssd-ci-control/current/venv/bin/python`: root-owned native Python
  3.10 environment containing the reviewed control dependencies; the installation
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

Use a repository-scoped runner registered **only** with the private controller
repository, carrying `self-hosted`, `Linux`, `X64`, `inferssd-device-ci`, `cloud-5090`.
Its process runs as the dedicated controller UID, with no sudo/worker privilege
or access to another repository's workflows. The concurrency group matches the
full cloud-5090 performance job; the device lock remains the final exclusion.

After reviewing those bytes and identities, dispatch on `main`:

```sh
gh workflow run integration-smoke.yml --repo wici-ai/inferssd-device-ci --ref main \
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

The report remains `purpose=integration-smoke`, `performance_acceptance=false`,
`benchmark_samples=0`. One successful model smoke demonstrates that the staged
build/start/inference/cleanup chain works. It is not the complete three-device CI
delivery, a performance comparison, a merge gate, or a formal benchmark sample.
