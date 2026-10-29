# Muse checkpoint workflow

Deploy `checkpoint.workflow.yml` as `.github/workflows/checkpoint.yml` on protected
`main` in the private controller repository, `wici-ai/inferssd-device-ci`. This
manual workflow compares the fixed legacy baseline (`LEGACY_REVISION` in
`bench/ci/reference.py`) with one explicitly reviewed candidate revision. It
builds both sides and runs one fixed Muse 512-prompt/256-decode request per side
in a single `StageSession` under the same device lease. It does not run 4K, retry a completion, publish an
App check, or replace the full three-device/two-model performance gate.

## Required deployment

The operator provisions the following before dispatch; this workflow downloads
no source, registers no runner and requests no App credential.

- `/opt/inferssd-ci-control/current/venv/bin/python`: root-owned Python 3.10 or
  later with the reviewed controller dependencies. `current` is a real protected
  directory, with no symlink ancestor, and cannot be written by either service UID.
- `/opt/inferssd-ci-control/current/bench/ci/staged_checkpoint.py` and all its
  imports: the installed reviewed coordinator. Its CLI is `--socket`, `--job-id`,
  `--config`, `--config-sha256`, `--output`.
- `/opt/inferssd-ci-integrations/<job_id>.json`: root-owned configuration readable
  by the controller UID. It declares `purpose=integration-checkpoint`,
  `device=cloud-5090`, the fixed `base.revision`, the reviewed `candidate.revision`,
  and `manifest_sha256`, together with the coordinator's model/build/launch and
  comparison binding. Dispatch pins the entire file's SHA-256. The coordinator
  checks the opened broker job against that exact manifest hash.
- `/run/inferssd-ci-broker/broker.sock`: reviewed root broker, with both source
  trees, model inputs and a fresh root-owned per-job manifest already staged.
  The existing device lock, STOP handling and worker isolation remain active.

Use a repository-scoped runner registered **only** with the private controller
repository, running as its dedicated controller UID without sudo/worker
privilege. Its labels are `self-hosted`, `Linux`, `X64`, `inferssd-device-ci`,
`cloud-5090`. The concurrency group `physical-cloud-5090` is shared with smoke and
full performance jobs; the original device lock remains the final exclusion.
The job also requires the exact repository, `refs/heads/main`, and a private
repository event. Do not register this runner with the public source repository.

## Dispatch and read the result

After staging and reviewing the configuration and both source identities:

```sh
gh workflow run checkpoint.yml --repo wici-ai/inferssd-device-ci --ref main \
  -f job_id=FULL_32_CHARACTER_JOB_ID \
  -f source_revision=FULL_40_CHARACTER_CANDIDATE_SHA \
  -f config_sha256=FULL_64_CHARACTER_CONFIG_SHA256
```

`source_revision` always means the candidate. The baseline is fixed, not a
workflow input. Inputs pass through environment variables and are validated
before use; arbitrary dispatch paths or commands are not accepted. A failed job
is not automatically retried. Preserve its evidence and stage a fresh reviewed
job ID if a corrected attempt is needed.

The upload step runs even after failure and retains the complete attempt under
`inferssd-checkpoint-<run_id>-<attempt>/`: controller logs, dispatch identities,
configuration, build receipts, raw SSE/timestamp ledgers, both arm reports,
runtime probes, stop/cleanup receipts, comparison and `checkpoint.json`.
Evidence that was never produced stays absent; it is not synthesized.

The job summary reads the actual `checkpoint.json` status and its comparison's
five metrics: first 128 decode tokens, second 128, overall decode, TTFT, and native
prefill. It displays base, candidate and their ratio. Missing or invalid reports
produce no benchmark verdict; missing metrics are shown as unavailable.
`performance_acceptance=true`, when actually reported, means only that this
single cell had valid evidence, no observed slowdown in those five metrics and
positive cleanup. `needs-review` records an observed slowdown; `failed` records
invalid or incomplete execution/evidence. Neither is reported as passing. Even
a passed sample is not statistical proof of no regression, the formal 512/4K
benchmark, or completion of the full device gate (`full_device_gate=false`).
