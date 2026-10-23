# Linux device stage broker

The trusted controller holds one JSONL Unix socket connection to
`/run/inferssd-ci-broker/broker.sock` for an entire job. The socket admits only
`inferssd-runner`. Candidate code runs as `inferssd-worker`; the trusted HTTP
client uses a third identity, `inferssd-client`. Neither worker inherits runner
credentials, coordinator descriptors, the device lease, or writable evidence.

`install.py` prints the installation plan by default. `--apply` is a root-only,
fresh installation: three non-login accounts, root-owned broker/state, original
device lock, and a **disabled** systemd service. It does not register a runner,
install models, run CUDA, or activate execution. A shared development device must
reuse its existing lock inode; do not create a second independent lock there.

Before activation, configure narrow root-owned software paths, exact GPU nodes,
resource limits, and accessible root-owned read-only source/model locations.
`probe.py --config /opt/inferssd-ci-broker/config.json` runs the real namespace,
credential denial, controller death/disconnect, STOP, original lock, and one
async build/freeze/server/client gate. It writes `isolation-proof.json` only if
all gates pass. The operator must separately create root-owned `ACTIVE` containing
the proof SHA256, then start the broker. Source/config changes invalidate proof;
never silently overwrite a previous proof or clear an unrelated STOP.

## Job manifest and protocol

The trusted provisioning/controller path places a root-owned manifest at
`/var/lib/inferssd-ci-broker/jobs/<32-lowercase-hex-job-id>/manifest.json`:

```json
{
  "schema_version": 1,
  "job_id": "0123456789abcdef0123456789abcdef",
  "sources": {"base": "/var/lib/inferssd-ci-inputs/base", "candidate": "/var/lib/inferssd-ci-inputs/candidate"},
  "models": {"model.gguf": "/var/lib/inferssd-ci-models/model.gguf"},
  "builds": {"base": "base", "candidate": "candidate"},
  "stages": {
    "base-build": {"kind": "build", "side": "base", "executables": ["/usr/bin/cmake"], "cwd": "/scratch/base", "gpu": false},
    "base-server": {"kind": "server", "side": "base", "executables": ["/build/base/bin/llama-server"], "cwd": "/scratch/base", "gpu": true}
  }
}
```

All messages include `schema_version: 1`, `operation`, and `job_id`.

- `job.open`: acquire original device lease and create fresh scratch; same live
  connection/job can call this idempotently. First `stage.start` also opens it.
- `stage.start`: additionally `stage_id`, `kind`, `argv`, `environment`, `cwd`,
  `timeout_s`. Executable, kind and cwd must match the trusted manifest. Each
  stage ID runs once. Response includes `handle`, `worker_pid`, `worker_uid`,
  `birth_identity`, `scope_identity`, `running`, `returncode`, and cleanup state.
- All stage followups require the exact `stage_id` and returned `handle`.
- `stage.poll` / `stage.wait`: wait optionally has `wait_s` (0–60). Both return
  status and bounded stdout. Prefer `stage.output` for complete raw logs.
- `stage.output`: `stream` (`stdout`/`stderr`), `offset`, `limit` (1–65536);
  returns `offset`, `next_offset`, `data_b64`, and `eof`.
- `stage.probe`: optional `port`; verifies exact worker birth, UID, cgroup and
  executable. Also returns actual `cmdline`, clean worker `environment`,
  `cpus_allowed_list`, `cgroup_limits` (`memory_max`, `memory_swap_max`,
  `pids_max`, as kernel strings), and raw `proc_limits` including memlock.
  A port requires one listener owned by that same pinned PID. Optional boolean
  `light: true` retains birth/UID/cgroup/listener checks while omitting full
  executable, mapping and launch evidence. Startup polling and HTTP admission
  use this mode so model memory allocation cannot stall repeated mapping reads;
  request-boundary evidence still uses the default full probe.
- `stage.client`: `request={port,method,path,body,timeout_s}`. Only fixed health
  and completion endpoints on private loopback are allowed. The separate-UID
  client timestamps each response line: `started_monotonic_ns`,
  `ended_monotonic_ns`, `chunks[{elapsed_ns,data_b64}]`. No worker timing summary
  is trusted. Ordinary EOF, timeout or an exited worker returns `status: 0` and
  `transport_error`, retaining the session for output/stop/finish. Changed or
  ambiguous process identity still refuses the operation. Response is bounded
  below the 1 MiB protocol limit.
- `stage.stop`: stop only that owned unit and confirm all descendants gone.
- `job.freeze`: use a successful build stage ID/handle. All workers must have
  stopped. Copy its scratch output to root-owned `/build/<side>`; retain the job
  lease. Normalize only links resolving to a regular unique file **inside the
  same tree**, recording link/target; reject escape, cycles, special files and
  multiply linked files. Normal builds stay shared/static as originally configured.
- `job.finish`: drain every stage, save trusted receipt, then release lease.
  Socket EOF cancels stages and follows the same cleanup path.

Sources mount at `/source/<side>`, models at `/models/<name>`, shared worker
scratch at `/scratch`, and sealed builds at `/build/<side>`. Tests and servers
require a sealed build. Request argv never executes in the root coordinator.

A root supervisor owns both systemd unit launch and cleanup, retaining a
separate lease descriptor. The trusted worker start barrier pins host PID/birth,
cgroup and network namespace **before** candidate exec. A killed controller
cannot release the lease while an unconfirmed worker survives. Unconfirmed
cleanup writes durable STOP and refuses subsequent admissions.

The worker retains the host cgroup path with a read-only `/sys` hierarchy, so
the engine can read its actual `memory.max`, current usage, and swap limit.
Creating a separate cgroup namespace against the host hierarchy would hide
that limit from adaptive memory accounting. The native probe checks both the
configured limit visibility and denied cgroup writes in its existing worker.

## Validation boundary

Run `python3 bench/ci/linux-sandbox/test_contract.py` for the focused local
artifact sealing, request admission and trusted HTTP timing tests. These tests
are not proof of Linux isolation. Native `probe.py` is model-free and produces
no benchmark samples. Real CUDA toolchain/driver operation and performance
comparison still require the independent trusted coordinator integration.

For DGX Spark, install with `--device dgx-spark --existing-lock ... --stop-path ...`
and the inventoried `--lock-device`, `--lock-inode`, `--operator-uid` and
`--operator-gid`. The installer preserves that existing operator-owned inode and
its owner. Its protected configuration binds the identity; runner/worker/client
identities cannot own or replace that path. Only the trusted broker receives
access to the coordination directory, never candidate namespaces. No independent
second device lease is created. Cloud provisioning can still create its first
root-owned lease when there is no pre-existing shared device lock.

On Ubuntu hosts that restrict unprivileged user namespaces, keep the global
restriction enabled. An operator may install an exact root-owned
`<install_root>/bin/bwrap` copy with a dedicated AppArmor `userns,` profile and
bind its path/SHA-256 in `namespace_launcher`. The broker validates that protected
identity. Do not grant the candidate a writable launcher or mount the broker
installation into its namespace. This path also requires bubblewrap
`--disable-userns --assert-userns-disabled`; the native negative payload verifies
that the candidate cannot create a nested user namespace or see the host helper. Validate the affected namespace/UID/native
cleanup gate once after the host-specific deployment. See Ubuntu's
[restricted namespace design](https://ubuntu.com/blog/ubuntu-23-10-restricted-unprivileged-user-namespaces).

Recent systemd versions can mask proc files in the parent namespace before
bubblewrap creates the nested PID namespace's procfs; Linux then refuses the
new proc mount. For the dedicated launcher only, the broker omits the redundant
parent `ProtectKernelTunables`/`ProtectKernelLogs` mounts. Bubblewrap still
creates a new PID/user/mount namespace with a private `/proc` and read-only
`/proc/sys`. CUDA may write its own thread names through `/proc/self/task/*/comm`;
`/sys` remains a read-only bind, capabilities remain dropped, and no host PID
namespace is exposed. Native proof checks writable own-task metadata, read-only
kernel tunables, private namespaces, hidden helper and denied nested user namespace.
