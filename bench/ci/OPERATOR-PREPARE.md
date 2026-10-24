# Operator preparation for two Linux devices

`operator_prepare.py` uses the preparer's existing `gh` login and SSH access.
It does not export a GitHub token, create a PAT, change the broker, register a
runner or contact the source API from a device. Candidate code only executes
later inside existing broker stages. The root staging helper copies source
objects as data and never executes them.

The local input directory must be trusted and outside a candidate checkout.
It contains reviewed device templates, model/settings identities and a profile:

```json
{
  "schema_version": 1,
  "controller_workflow_sha": "FULL_PRIVATE_CONTROLLER_MAIN_SHA",
  "devices": {
    "cloud-5090": {
      "template": "cloud-template.json",
      "manifest": "cloud-manifest-template.json",
      "control_revision": "FULL_REVIEWED_CONTROL_SOURCE_SHA",
      "control_manifest_sha256": "INSTALLED_CONTROL_MANIFEST_SHA256",
      "lock_identity": [0, 0],
      "gpu_uuid": "GPU-ACTUAL_UUID",
      "min_free_bytes": 8589934592,
      "rootstage_files": {
        "settings-deepseek.ssd": "deepseek.ssd",
        "settings-muse.ssd": "muse.ssd"
      },
      "ssh": {
        "argv": ["/usr/bin/ssh", "-T", "-a", "-o", "BatchMode=yes", "REVIEWED_ENDPOINT"],
        "sudo_password_required": false,
        "root_argv": ["/usr/bin/python3", "-I", "/opt/inferssd-ci-control/current/bench/ci/operator_stage.py"]
      }
    },
    "dgx-spark": {"EXPLICIT_REVIEWED_PROFILE": "same required fields; no defaults"}
  }
}
```

The placeholder values above are not executable configuration. Both device
profiles are required, but `--execution-devices` can choose one or both Linux
devices. An optional per-device `job_id` must be fresh, distinct and 32 lowercase
hex characters; otherwise preparation generates one. Old template job/source
identities are replaced with the exact new identities. Base remains provenance
metadata; this mode stages/builds only candidate. Template resource/model values
are retained. Settings payload hashes must match the declarations.

Prepare a reviewable package without connecting to a device:

```sh
python3 -I bench/ci/operator_prepare.py prepare \
  --source /trusted/local/repository --base FULL_BASE --head FULL_HEAD \
  --candidate FULL_CANDIDATE --inputs /trusted/operator/inputs.json \
  --output /trusted/operator/fresh-output --execution-devices cloud-5090 \
  --gh /path/to/existing/gh
```

The authenticated source Git API must match each complete local commit/tree and
its parents. Candidate must equal head with base an ancestor, or merge precisely
base/head. The package preserves the complete binary diff and both names of a
rename. A single batch Git blob reader exports every regular tracked source
file, including files marked `export-ignore`; no checkout/filter/hook runs.

`envelope.json` retains the canonical impact plan, including required Mac and
performance cells. Separate `execution_devices` and `validation_mode` limit this
run to Linux functional/numerical work. The complete workflow input must fit
60 KiB; exceeding that bound refuses without truncation. No 256-token request
is authorized by this mode, and its report cannot claim complete impact coverage
or PR check publication.

Before enabling root staging, review and install each generated
`DEVICE/policy.review.json` as the local root-owned, nonwritable
`/opt/inferssd-ci-control/operator-policy.json`. Install the exact reviewed
controller export, including `operator_stage.py`, first. This is an administrator
bootstrap operation, separate from candidate preparation. The helper cannot
replace its policy. It binds the device, broker config and original lock inode,
GPU UUID, control manifest, exact resource envelope, model mounts/identities,
settings hashes and existing Spark admission/proof. Extra host mappings such as
runner home or `/etc/shadow` are refused. All new destinations are fixed CI
prefixes with fresh exclusive writes. Existing model bytes/mounts, broker config,
lock and STOP are not changed.

On a nonroot host, install the reviewed `inferssd-ci-stage.socket` and
`inferssd-ci-stage@.service` templates with the actual permitted peer UID/group.
The same fixed Python command above becomes a local client of
`/run/inferssd-ci-stage.sock`; use `sudo_password_required: false` and omit
`sudo_mode`. The root service accepts only that UID, a bounded SHA header and
the plain tar, then invokes the same policy-bound receiver. Its fixed service
command and owner-controlled socket do not grant arbitrary root commands.
Keep the socket outside candidate worker mounts. Preserve vendor sudo/SSH
policy; do not change its ordering to obtain passwordless execution. Review
the forced SSH command as well: it must accept this exact client command.

For an existing permitted and separately reviewed narrow NOPASSWD entry, a profile uses
`sudo_mode: "nopasswd"`, `sudo_password_required: false`, and the fixed root argv
`["/usr/bin/sudo", "-n", "/usr/bin/python3", "-I",
"/opt/inferssd-ci-control/current/bench/ci/operator_stage.py"]`. It does not permit
arbitrary Python/bash files or a caller-selected policy path. Bootstrap/debug
can instead use `sudo_password_required: true` and prefix the fixed Python
command with `"/usr/bin/sudo", "-S", "-p", ""`; `stage/run --sudo-prompt` reads
one password hidden into memory and writes its line only to sudo's SSH stdin,
before the plain tar. It is absent from argv, environment, packages and logs.

Stage and dispatch the prepared package:

```sh
python3 -I bench/ci/operator_prepare.py stage --prepared /trusted/operator/fresh-output
python3 -I bench/ci/operator_prepare.py dispatch --prepared /trusted/operator/fresh-output \
  --gh /path/to/existing/gh
```

`run` accepts the preparation arguments and performs prepare → stage → dispatch.
It is the same backend for a trusted review watcher; it is not a second
authorization mechanism. The sender uses non-PTY SSH and a plain tar stream,
matching the root receiver's `r|` contract. Root holds the original lease during
staging, rejects STOP, verifies GPU UUID and absence of compute work, checks the
entire installed control manifest, reads model stat/RO mount metadata, and
verifies every extracted Git blob before sealing source readonly. It installs
only the reviewed candidate config/settings and job manifest, then returns an
identity-bound root receipt.

Dispatch requires all chosen root receipts and an unchanged local account and
private controller main. It sends only `prepared_envelope` to the fixed
`device-performance.yml` workflow. A durable exclusive claim precedes SSH stage
or the one dispatch POST. Failed or ambiguous attempts remain recorded and
cannot be automatically retried with the same prepared directory/job. An exact
returned run ID is required; there is no heuristic run search. The CLI records
the actual parent run and prints its ID/URL. The existing hosted controller
owns child dispatch, parent cancellation, evidence collection and report-only
Actions results; a successful cancellation API call never proves cleanup.
The dispatch request uses GitHub's documented
[`return_run_details` contract](https://docs.github.com/en/rest/actions/workflows#create-a-workflow-dispatch-event).

The local functions are `prepare(args, api=None)`,
`stage(args, sender=send_stage, password=None)` and `dispatch(args, api=None)`.
`dispatch` returns the actual identity-checked run dictionary. Tests use small
temporary Git repositories and synthetic root filesystems/transports. They do
not establish real device isolation, native math, a live dispatch or deployment.
