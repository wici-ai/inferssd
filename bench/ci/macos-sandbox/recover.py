#!/usr/bin/python3
"""Fixed root SSH-context broker recovery; never executes candidate code.

No launchd/TCC policy is changed. A reachable unmanaged, busy or mismatched
broker is refused, never killed. The original device lease is only opened and
flocked, never created, replaced, chowned or unlinked.
"""
import errno
import fcntl
import json
import os
from pathlib import Path
import socket
import stat
import sys
import time
import uuid

# Isolated Python deliberately omits the script directory from sys.path. Only
# the fixed root-protected installation may bootstrap its sibling modules.
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensure", action="store_true", required=True)
    parser.parse_args()
    fixed = Path("/Library/InferSSD-CI/control/recover.py")
    if Path(__file__).absolute() != fixed:
        raise RuntimeError("recovery must use the fixed installed entry")
    for item in (fixed, *fixed.parents):
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
            raise RuntimeError("unprotected recovery entry ancestor")
    sys.path.insert(0, str(fixed.parent))

import supervisor as core


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def management_lock():
    path = core.BASE / "private/broker-recovery.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_uid == 0 and info.st_nlink == 1
                and not info.st_mode & 0o077, "unprotected broker management lock")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except BaseException:
        os.close(fd)
        raise


def socket_identity(path, config):
    info = path.lstat()
    require(stat.S_ISSOCK(info.st_mode) and info.st_uid == 0
            and info.st_gid == config["runner_gid"]
            and stat.S_IMODE(info.st_mode) == 0o660, "unprotected broker socket")
    return (info.st_dev, info.st_ino)


def status(config, timeout=2):
    """Only absent/refused means stale; timeout/unknown protocol remains an error."""
    import executor
    path = core.BASE / "control/executor.sock"
    try:
        socket_identity(path, config)
    except FileNotFoundError:
        return None
    with socket.socket(socket.AF_UNIX) as peer:
        peer.settimeout(timeout)
        try:
            peer.connect(str(path))
        except OSError as error:
            if error.errno == errno.ECONNREFUSED:
                return None
            raise
        require(executor.peer_identity(peer) == (0, 0), "broker peer is not root")
        peer.sendall(b'{"schema_version":1,"operation":"broker.status"}\n')
        with peer.makefile("rb") as stream:
            line = stream.readline(65537)
        require(line.endswith(b"\n") and len(line) <= 65536, "invalid broker status frame")
        def unique(pairs):
            result = {}
            for key, value in pairs:
                require(key not in result, "duplicate broker status key")
                result[key] = value
            return result
        result = json.loads(line, object_pairs_hook=unique)
        expected = {"schema_version": 1, "status": "ready", "protocol": "mac-executor-v1",
                    "config_sha256": core.sha(core.BASE / "control/config.json"),
                    "installed_files": config["installed_files"],
                    "lock_dev_inode": config["lock_dev_inode"],
                    "worker_uid": config["worker_uid"], "runner_uid": config["runner_uid"]}
        require(isinstance(result, dict) and set(result) == set(expected) | {"broker_birth"},
                "unmanaged or incompatible broker")
        require(all(result[key] == value for key, value in expected.items()),
                "running broker identity differs; explicit idle migration required")
        birth = result["broker_birth"]
        require(isinstance(birth, dict) and type(birth.get("pid")) is int
                and birth.get("uid") == 0 and executor.birth(birth["pid"]) == birth,
                "broker PID/birth identity differs")
        return result


def validate_idle(config):
    original_lock(config)
    lease = core.lease(config)
    try:
        require(not core.members(config["worker_uid"]), "worker UID is not empty")
        original_lock(config)
        return lease
    except BaseException:
        os.close(lease)
        raise


def original_lock(config):
    stop = Path(config["stop"])
    require(not (stop.exists() or stop.is_symlink()), "STOP is present")
    fd = os.open(config["lock"], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and [info.st_dev, info.st_ino] == config["lock_dev_inode"],
                "original lock inode differs")
    finally:
        os.close(fd)


def spawn(config, manager, lease):
    """Detach from SSH while closing its archive, credentials and control FDs."""
    log = core.BASE / "private" / ("broker-recovery-" + uuid.uuid4().hex + ".log")
    logfd = os.open(log, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        pid = os.fork()
    except BaseException:
        os.close(logfd)
        raise
    if pid == 0:
        try:
            os.setsid()
            null = os.open("/dev/null", os.O_RDONLY)
            os.dup2(null, 0)
            os.dup2(logfd, 1)
            os.dup2(logfd, 2)
            for value in os.listdir("/dev/fd"):
                fd = int(value)
                if fd > 2 and fd not in (manager, lease):
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            # Do not retain SSH environment, agent socket or loader settings.
            os.environ.clear()
            os.environ.update(PATH="/usr/bin:/bin", LANG="C", LC_ALL="C")
            import executor
            executor.serve(config, management_fd=manager, bootstrap_lease_fd=lease)
        except BaseException as error:
            os.write(2, (type(error).__name__ + ": " + str(error) + "\n").encode())
            os._exit(1)
        os._exit(0)
    os.close(logfd)
    return pid, str(log)


def ensure():
    config = core.config()  # root/Darwin + installed code/tool hashes
    require(config.get("executor", {}).get("enabled") is True, "executor is disabled")
    original_lock(config)
    current = status(config)
    if current is not None:
        original_lock(config)
        return {"action": "reused", **current}
    manager = management_lock()
    lease = None
    try:
        # Recheck after serializing with every managed broker/guardian.
        current = status(config)
        if current is not None:
            original_lock(config)
            return {"action": "reused", **current}
        lease = validate_idle(config)
        path = core.BASE / "control/executor.sock"
        if path.exists() or path.is_symlink():
            before = socket_identity(path, config)
            require(status(config) is None, "broker became live during recovery")
            require(socket_identity(path, config) == before, "broker socket changed during recovery")
            path.unlink()
        pid, log = spawn(config, manager, lease)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            waited, _ = os.waitpid(pid, os.WNOHANG)
            require(waited == 0, "broker exited during recovery; retained log: " + log)
            current = status(config, timeout=.5)
            if current is not None:
                require(current["broker_birth"]["pid"] == pid, "another broker answered recovery")
                original_lock(config)
                return {"action": "started", "log": log, **current}
            time.sleep(.1)
        raise RuntimeError("broker readiness unconfirmed; do not retry/kill blindly; retained log: " + log)
    finally:
        if lease is not None:
            os.close(lease)
        os.close(manager)


if __name__ == "__main__":
    print(json.dumps(ensure(), sort_keys=True))
