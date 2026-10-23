#!/usr/bin/python3
"""Render a reviewable installation plan; --apply is an explicit root action."""

import argparse
import grp
import hashlib
import json
import os
from pathlib import Path
import pwd
import subprocess
import stat

FILES = ("contract.py", "executor.py", "guardian.py", "daemon.py", "client.py", "worker_probe.py", "probe.py",
         "stages.py", "worker_entry.py", "http_client.py", "stage_probe.py")
INSTALL = Path("/opt/inferssd-ci-broker")
STATE = Path("/var/lib/inferssd-ci-broker")
SERVICE = """[Unit]
Description=InferSSD trusted device stage broker
After=local-fs.target

[Service]
Type=simple
User=root
ExecStart=/usr/bin/python3 -I /opt/inferssd-ci-broker/daemon.py --config /opt/inferssd-ci-broker/config.json
Restart=on-failure
RestartSec=2
# Independent root guardians retain leases and stop their exact worker units
# when the broker dies. Do not kill those guardians with the broker process.
KillMode=process
RuntimeDirectory=inferssd-ci-broker
RuntimeDirectoryMode=0755
UMask=0077
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/var/lib/inferssd-ci-broker /run/inferssd-ci-broker
PrivateTmp=yes
# Trusted clients join a worker's network namespace before dropping UID.
# They need AF_INET there; candidate stages separately have PrivateNetwork.
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6

[Install]
WantedBy=multi-user.target
"""


def account(name, home):
    try:
        user = pwd.getpwnam(name)
    except KeyError:
        subprocess.run(["/usr/sbin/useradd", "--system", "--user-group", "--home-dir", str(home),
                        "--create-home", "--shell", "/usr/sbin/nologin", name], check=True)
        user = pwd.getpwnam(name)
    memberships = [group.gr_name for group in grp.getgrall() if name in group.gr_mem]
    if user.pw_uid == 0 or user.pw_gid == 0 or memberships:
        raise ValueError("dedicated accounts must have no supplementary groups")
    if Path(user.pw_dir) != home or user.pw_shell not in ("/usr/sbin/nologin", "/bin/false"):
        raise ValueError("existing account is not the reviewed non-login identity")
    return user


def write(path, data, mode):
    if path.exists() or path.is_symlink():
        raise FileExistsError("refuse to replace installed material: " + str(path))
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data.encode() if isinstance(data, str) else data)
        stream.flush()
        os.fsync(stream.fileno())


def main(args):
    source = Path(__file__).resolve().parent
    existing = args.existing_lock
    if args.device == "dgx-spark" and existing is None:
        raise ValueError("DGX deployment must preserve the reviewed original device lock")
    if existing:
        if not existing.is_absolute() or ".." in existing.parts or existing.is_symlink():
            raise ValueError("original lock must be an absolute regular path")
        info = existing.lstat()
        if (not stat.S_ISREG(info.st_mode) or [info.st_dev, info.st_ino] != [args.lock_device, args.lock_inode]
                or info.st_uid != args.operator_uid or info.st_gid != args.operator_gid):
            raise ValueError("original lock differs from explicit reviewed identity")
        if not args.stop_path or args.stop_path.parent != existing.parent:
            raise ValueError("explicit original STOP path beside existing lease required")
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in FILES}
    plan = {"schema_version": 1, "install_root": str(INSTALL), "state_root": str(STATE),
            "controller": "inferssd-runner", "worker": "inferssd-worker", "client": "inferssd-client", "files": hashes,
            "device": args.device, "existing_lock": str(existing) if existing else None,
            "actions": ["create three unprivileged non-login accounts", "install root-owned reviewed files",
                        "preserve existing coordination lease or create first cloud lease", "install disabled broker service"],
            "not_activated": ["no runner registration", "no ACTIVE proof binding", "no models or CUDA installation", "no physical benchmark"]}
    if not args.apply:
        print(json.dumps(plan, indent=2))
        return 0
    if os.geteuid() != 0 or os.uname().sysname != "Linux":
        raise PermissionError("installation requires explicit root on Linux")
    if INSTALL.exists() or STATE.exists():
        raise FileExistsError("fresh install only; review an explicit upgrade instead of overwriting state")
    runner = account("inferssd-runner", Path("/var/lib/inferssd-runner"))
    worker = account("inferssd-worker", Path("/var/lib/inferssd-worker"))
    client = account('inferssd-client', Path('/var/lib/inferssd-client'))
    if runner.pw_uid == worker.pw_uid or runner.pw_gid == worker.pw_gid:
        raise ValueError("controller/worker identities must differ")
    if len({user.pw_uid for user in (runner, worker, client)}) != 3 or len({user.pw_gid for user in (runner, worker, client)}) != 3:
        raise ValueError('three dedicated identities required')
    for user in (runner, worker, client):
        os.chmod(user.pw_dir, 0o700)
    INSTALL.mkdir(mode=0o755)
    STATE.mkdir(mode=0o711)
    (STATE / "stages").mkdir(mode=0o711)
    (STATE / 'jobs').mkdir(mode=0o711)
    for name in FILES:
        write(INSTALL / name, (source / name).read_bytes(), 0o755 if name in ("client.py", "probe.py") else 0o644)
    (INSTALL / "etc").mkdir(mode=0o755)
    write(INSTALL / "etc/passwd", f"worker:x:{worker.pw_uid}:{worker.pw_gid}:worker:/scratch/home:/usr/sbin/nologin\n", 0o644)
    write(INSTALL / "etc/group", f"worker:x:{worker.pw_gid}:\n", 0o644)
    write(INSTALL / "etc/nsswitch.conf", "passwd: files\ngroup: files\nhosts: files\n", 0o644)
    write(INSTALL / "etc/hosts", "127.0.0.1 localhost\n::1 localhost\n", 0o644)
    write(INSTALL / "etc/resolv.conf", "", 0o644)
    lock = existing or STATE / "device.lock"
    if existing is None:
        write(lock, "", 0o600)
    info = lock.stat()
    software = [str(p) for p in (Path("/usr/bin"), Path("/usr/sbin"), Path("/usr/include"), Path("/usr/lib"), Path("/usr/lib64"), Path("/usr/libexec"), Path("/usr/share")) if p.is_dir() and not p.is_symlink()]
    config = {"schema_version": 1, "activation_ready": False, "device": args.device, "platform": "linux",
              "install_root": str(INSTALL), "state_root": str(STATE),
              "controller_uid": runner.pw_uid, "controller_gid": runner.pw_gid,
              "worker_uid": worker.pw_uid, "worker_gid": worker.pw_gid,
              'client_uid': client.pw_uid, 'client_gid': client.pw_gid,
              "lock_path": str(lock), "lock_identity": [info.st_dev, info.st_ino],
              "stop_path": str(args.stop_path or STATE / "STOP"), "readonly_system": software,
              "gpu_devices": [], "gpu_uuid": "", "memory_max": 1024**3, "tasks_max": 256,
              "log_max": 64 * 1024**2, 'build_max_bytes': 32 * 1024**3}
    if existing:
        config["coordination_owner"] = {"uid": args.operator_uid, "gid": args.operator_gid}
    from contract import coordination_paths
    coordination_paths(config)
    write(INSTALL / "config.json", json.dumps(config, indent=2) + "\n", 0o644)
    write(INSTALL / "install-receipt.json", json.dumps(plan, indent=2) + "\n", 0o644)
    unit = Path("/etc/systemd/system/inferssd-ci-broker.service")
    service = SERVICE
    if existing:
        service = service.replace("ProtectHome=yes", "ProtectHome=read-only").replace(
            "ReadWritePaths=/var/lib/inferssd-ci-broker /run/inferssd-ci-broker",
            "ReadWritePaths=/var/lib/inferssd-ci-broker /run/inferssd-ci-broker " + str(existing.parent))
    write(unit, service, 0o644)
    subprocess.run(["/usr/bin/systemctl", "daemon-reload"], check=True)
    print(json.dumps({"installed": True, "service_started": False, "runner_registered": False,
                      "config": str(INSTALL / "config.json"), "negative_probe_required": True}))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--device", choices=("cloud-5090", "dgx-spark"), default="cloud-5090")
    parser.add_argument("--existing-lock", type=Path)
    parser.add_argument("--stop-path", type=Path)
    parser.add_argument("--lock-device", type=int)
    parser.add_argument("--lock-inode", type=int)
    parser.add_argument("--operator-uid", type=int)
    parser.add_argument("--operator-gid", type=int)
    raise SystemExit(main(parser.parse_args()))
