# macOS device stages

This is a root-installed, default-disabled stage executor. No Actions runner is
registered by the installer. The target is the inventoried M5 Max, and the
installer preserves the original device lock inode and STOP path. The portable
tests establish only request and file-copy behavior; native isolation and Metal
execution must pass before activation.

The earlier installed package's native C++/Metal and isolation evidence remains
in the private archive under its original hashes. This public template now
requires explicit reviewed installer target inputs and uses a public address for
the no-payload external-connect denial probe. These adjusted template bytes have
not received a new native pass; the package manifest marks that limitation.

## Installation boundary

Review `install.py` and the package hashes, then invoke its explicit installation
flag through a root terminal. Supply `--reviewed-hostname`, `--personal-home`,
`--personal-uid`, `--lock-path`, `--lock-device` and `--lock-inode` from the private,
reviewed inventory. No target identity is guessed or defaulted. Installation
checks the actual hostname, home ownership/mode and original lock device/inode
before changing any accounts or files. Never substitute a newly created lock.
It creates `/Library/InferSSD-CI` and two hidden,
disabled, false-shell accounts: runner 62001 and worker 62002, each with a distinct
primary group. macOS may add implicit directory-service memberships such as
everyone/localaccounts; none may be wheel/staff/admin. Every actual worker and
future runner service explicitly calls setgroups before dropping privilege;
Darwin still reports implicit nonprivileged groups. Native proof records actual
credentials and must prove protected-resource denial, without falsely claiming
a primary-only group list. It never changes the personal account/home permissions, creates
sudoers entries, or starts a service. Installation refuses an existing directory
or account collision rather than modifying it implicitly. The only resume mode
requires the exact root-owned empty-layout/partial-disabled-accounts checkpoint
and supplied SHA-256, binds this package manifest, verifies all directory and
account identities, and never resets existing account attributes. The normal system
`/Applications` ancestor is accepted only as root:admin 0775 without ACLs;
concrete Xcode descendants remain root-protected and neither service account may
belong to admin. Actual tool ancestry and account memberships enter the receipt.

The inherited `verify.py` performs one meaningful negative set, its child
inheritance check, real C++ and Metal computations, disconnect cleanup, original
lease reacquisition and STOP refusal. It deliberately cannot certify the newer
production protocol, model staging or bounded server profile by implication.
The earlier installed M5 package exercised these Metal permissions successfully.
Concrete runtime corrections grant only root-directory data, explicit tool/system
ancestor metadata, and the exact read-only libinfo lookup; protected content stays
denied. Compiler/SDK paths are frozen directly instead of invoking Xcode shims.
There is no broad-permission retry.

`stage_models.py` APFS-clones complete-hash-bound model inputs into a new,
root-owned read-only directory. It rejects a failed clone rather than falling
back to a hardlink or unexpectedly consuming a terabyte. Original personal files
keep their owner and inode. Any incomplete staging directory remains private.

## Root job manifest

The privileged preparer, never candidate code, creates this protected file:

`/Library/InferSSD-CI/private/job-manifests/<job_id>.json`

```json
{
  "schema_version": 1,
  "job_id": "0123456789abcdef0123456789abcdef",
  "input_roots": [
    "/Library/InferSSD-CI/inputs/review-123/base",
    "/Library/InferSSD-CI/inputs/review-123/candidate"
  ],
  "model_roots": ["/Library/InferSSD-CI/models/verified-model-set"],
  "stage_environments": {
    "base-server": {"GGML_MOE_CACHE_BYTES": "13976514560"}
  }
}
```

All entries must be root-owned directories in the dedicated staging roots.
Descendants must be read-only regular files/directories; symlinks and special
files are refused. The preparer binds sources/configuration and model hashes
before writing this manifest. The executor returns its SHA-256 in job identity.
Only the manifest's inputs and model roots enter that job's Seatbelt profile.

## Coordinator protocol

The installed invocation is:

`<config.tools.python> -I /Library/InferSSD-CI/control/client.py`

Use the frozen real Xcode Python interpreter, not the `/usr/bin/python3` shim;
first use of the shim can start per-user LaunchServices helpers.

It is a persistent JSONL bridge to the root Unix broker socket. The broker checks
its actual peer UID/GID, not a field supplied by the caller. An independent root guardian, connected to the broker by a private lifeline,
holds the original device lease; broker death closes the lifeline and triggers
cleanup. Candidate processes never inherit that socket or lease. One connection holds
one original device lease across every stage and both sides. All requests have
`schema_version: 1`, `operation` and `job_id`.

- `job.open` creates the new job and returns `job_root`, `scope_identity`,
  `worker_uid`, `lock_dev_inode`, manifest hash and read-only input/model roots.
- `stage.start` also takes `stage_id`, `kind` (`build`, `test`, `server`, `probe`),
  `argv`, `environment`, absolute `cwd`, and `timeout_s`. CWD is inside this job;
  executable is absolute. Environment permits locale/timezone/build parallelism
  and exact per-stage key/value pairs from the root manifest, allowing legacy
  baseline settings. Loader, shell startup and Python injection variables remain
  forbidden even in that manifest. HOME/TMPDIR/PATH are fixed by the broker.
  Job IDs must be 32 lowercase hex characters. It returns
  `handle`, `worker_pid`, `worker_uid`, kernel `birth_identity`, `scope_identity`,
  `running`, `returncode` and `cleanup_confirmed`.
- Every stage follow-up requires the same `job_id`, `stage_id`, `handle`.
  `stage.poll` reports state; `stage.wait` accepts `wait_s` at most 30;
  `stage.stop` cleans the complete dedicated UID, including detached children.
  The guardian also retires that worker UID’s launchd user domain, so macOS XPC
  helpers do not respawn after an apparently empty process inventory.
- `stage.output` takes `stream` (`stdout`/`stderr`), byte `offset`, `limit` at most
  65536. It returns `offset`, `next_offset`, `data_b64`, and `eof`. Trusted logs
  remain root-owned and cannot be modified by the candidate. Workers receive
  pipes only; the guardian drains them into bounded root-owned logs. No private
  evidence file descriptor reaches the candidate.
- `stage.probe` returns kernel PID/birth identity, actual dedicated-UID members,
  aggregate RSS and trusted `/usr/sbin/lsof` output. `stage.client` verifies the
  exact owned listener at `127.0.0.1:18080` and returns its URL and listener PIDs.
  HTTP/SSE timing and comparison run in the trusted coordinator.
- `job.freeze` binds the completed successful build `stage_id` and `handle`, takes `source` (inside job work) and unique `name` (use `base` or
  `candidate`). It requires an empty worker UID, makes a new root-owned read-only
  byte copy, and returns path, manifest path/hash and file count. Internal file
  symlinks resolve only within the same build tree and become regular copies with
  original link/target recorded; directory links, escapes, cycles, special files
  and multiply linked files are refused. No untrusted inode is chowned.
- `job.finish` cleans workers and releases the lease. EOF/controller death does
  the same. Unknown cleanup creates STOP, retains the lease until UID emptiness
  is proved and permanently fails that job.

The root broker is `executor.py`; it refuses to start while
`config.executor.enabled` is false. Enabling remains a separate operator step
only after actual native gates and adapter verification. Mac memory/process
checks are sampled safeguards, not a hard UMA/swap isolation claim.

## Focused local checks

```sh
python3 -m unittest discover -s bench/ci/macos-sandbox/tests -v
```

The tests cover safe regular/dylib sealing, hardlink/special-file/escape refusal,
byte limits, stage/job handle binding, path constraints and environment/deadline
validation. They do not launch a GPU, certify a sandbox or alter a device.

`verify_broker.py` is one fixed native SIGKILL check: a private in-memory config
copy starts the broker, an authenticated runner starts a sleeping sandbox worker,
the probe confirms lease exclusion, kills only its broker PID, and requires the
independent guardian to clean the worker before lease reacquisition. The installed
configuration remains disabled throughout. This proof does not replace model
staging or the still-required positive/negative fixed-port server check.
