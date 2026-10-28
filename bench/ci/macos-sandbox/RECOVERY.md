# SSH-context broker recovery

The fixed, root-installed entry is:

```
<config.tools.python> -I /Library/InferSSD-CI/control/recover.py --ensure
```

The trusted Mac root stager calls this before taking the original device lease.
`ensure()` is also available to that root-owned stager. It accepts no candidate
command, configuration, path or enablement request. The caller must be in the
already authorized SSH context: opening the actual original lock tests current
TCC access. Neither `SSH_CONNECTION` nor launchd/runner status is used as a
permission or device-idle proof.

The initial deployment needs root installation of the reviewed control files,
their `config.installed_files` hashes, and a dedicated restricted SSH key whose
forced command allows only the fixed root stager. Any sudo grant must name that
fixed installed interpreter/script/argument shape; no general Python, shell,
user-supplied script, personal password storage or candidate credential is
required. Recovery does not install keys, sudo rules or services itself. The
original Documents lock and STOP paths stay unchanged. No TCC, Full Disk Access,
sshd or personal-home policy is changed.

`broker.status` is a root-only operation and does not create a job or acquire the
device lease. It returns the startup configuration SHA256, installed-file table,
original lock identity, dedicated UIDs and broker kernel PID/birth identity. A
root caller cannot use this connection to open a candidate job. Recovery reuses
only an exact matching response. A busy socket, timeout, unexpected response,
changed control/configuration or unknown legacy broker is an error, not a
reason to stop or replace a process. Because the existing executor serializes
connections, a long active job can make status time out; the fresh stage fails
closed and does not interfere with that job.

For a missing socket or `ECONNREFUSED`, recovery acquires the separate
root-protected management lock, rechecks the socket, then obtains the original
lease and proves the worker UID empty and STOP absent. Only a root-owned 0660
socket with the configured runner group may be unlinked, and only after its
identity is checked again. Recovery never creates, replaces, changes ownership
of or unlinks the original device lock. A new broker inherits the management
lock and short startup lease, but no SSH/archive/agent/control descriptors or
environment. It releases the startup lease after listening; each actual job
continues to use the existing independent guardian and original lease.

The broker and its forked guardians retain the management lock. Killing a
broker therefore does not permit another broker to start while a guardian is
still cleaning a worker. Recovery never kills a broker, guardian or worker.
Startup failure retains its private log and does not retry a native operation.
An active broker's configuration cannot be silently refreshed in place.

The previously installed detached broker has no `broker.status` contract. Its
one-time migration requires a separate reviewed root operation that proves the
exact PID/birth, worker emptiness, original-lease availability and STOP absence
before retiring it. A missing launchctl service or an existing socket is
insufficient. After this migration each fresh authorized stage can recover a
missing broker through the fixed SSH entry, including after reboot. Native
TCC/Metal, disconnect and fresh human-authorized `/ci test` validation remain
required before claiming unattended device CI is active.

The reviewed `operator_entry.py` is installed with identical bytes at the fixed
`control/operator-ssh-entry` and `control/operator-root-entry` paths. Its frozen
Xcode Python shebang uses isolated mode. The first admits only the exact fixed
stager argv and one lowercase SHA256 from `SSH_ORIGINAL_COMMAND`, then invokes
`sudo -n` on the second. The second requires root and revalidates that sole digest
before replacing itself with the isolated fixed stager. Both clear inherited
environment. A dedicated SSH key must force the first path and disable forwarding,
PTY, agent and user startup hooks. A narrow sudo rule grants only the second
root-owned executable; no interpreter or caller-provided script is granted.
Installation of these files alone grants no SSH or sudo access.

Local tests exercise identity/refusal/stale-socket contracts without starting a
broker, acquiring a device lease, changing a Mac or running a model. They are not
native isolation or TCC evidence. The Mac memory policy samples worker physical
footprint, host free memory and process count; RSS is diagnostic only. It is not
Linux cgroup hard UMA/swap isolation. Normal job cleanup still relies on the existing dedicated-UID
guardian, not a recovery response or runner online status.
