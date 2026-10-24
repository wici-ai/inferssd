# Fixed read-only model mounts

These operator templates restore already-reviewed file binds before the trusted
broker starts. They do not discover models, accept runner parameters, copy
weights, or hash full model payloads.

A root operator replaces the `@@...@@` values before installation: absolute
manifest/source/target paths, source UID/GID, and a Python literal containing
the exact shard filenames. Use root-protected paths without shell or systemd
metacharacters. The root-owned manifest has a `files` list with `name`, `source`,
`path` (the target), `inode`, `bytes`, and `mtime_ns`. The manifest and target
directory ancestors must be root-owned and not group/world writable. Existing
unmounted targets must be empty, root-owned, read-only regular files.

Install the reviewed rendered helper as
`/opt/inferssd-ci-broker/ensure-readonly-models.py`, the rendered service in
`/etc/systemd/system/`, and `readonly-models.conf` under
`/etc/systemd/system/inferssd-ci-broker.service.d/`. Preserve the original device
lease and wait until it is available before changing mount/service state.
Enable the oneshot service, reload systemd, then start the broker through its
normal dependency. Retain the rendered files and hashes in deployment evidence.

The helper checks the exact source identity and existing read-only binds. It
creates only missing approved binds and makes them read-only; a failed remount
unmounts only the newly created bind. The source filesystem must already have a
verified boot mount mechanism. `RequiresMountsFor` alone is not evidence that an
external or manually mounted source will return after reboot.

After installation, verify service dependencies, read a GGUF header and deny a
write from the actual candidate namespace, and confirm cleanup/original lease
release. No additional model generation is needed. Record reboot behavior as
unverified unless an actual reboot test was performed.
