"""Root-only stage execution. Candidates never receive coordinator descriptors."""

import fcntl
import json
import os
from pathlib import Path
import select
import socket
import stat
import subprocess
import sys
import time
import uuid

from contract import UNIT, protected

CLEAN_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}


def show(unit):
    if not UNIT.fullmatch(unit):
        raise ValueError("unit is not owned")
    result = subprocess.run(["/usr/bin/systemctl", "show", unit, "--no-pager",
        "--property=LoadState,ActiveState,SubState,MainPID,ControlGroup,MemoryMax,MemorySwapMax,User,Group,PrivateUsers,PrivateNetwork,NoNewPrivileges,Result"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=CLEAN_ENV, timeout=10)
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def cleanup(unit, cgroup="", timeout=12):
    """Signal only the fresh exact unit; never match binary names or old PIDs."""
    if not UNIT.fullmatch(unit):
        raise ValueError("unit is not owned")
    own = "/system.slice/" + unit
    if cgroup and (not (cgroup == own or cgroup.startswith(own + "/")) or ".." in Path(cgroup).parts):
        raise ValueError("unexpected stage cgroup")
    subprocess.run(["/usr/bin/systemctl", "stop", unit], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, env=CLEAN_ENV, timeout=timeout)
    deadline = time.monotonic() + timeout
    while True:
        state = show(unit)
        control = state.get("ControlGroup") or cgroup
        path = Path("/sys/fs/cgroup") / control.lstrip("/") if control else None
        populated = False
        if path and path.exists():
            # Check the entire owned subtree, including descendants created by
            # a candidate. ProtectControlGroups prevents host-side delegation.
            for entries in path.rglob("cgroup.procs"):
                populated |= bool(entries.read_text().strip())
        gone = (state.get("LoadState") == "not-found" or
                state.get("ActiveState", "inactive") in ("inactive", "failed")) and not populated
        if gone:
            return {"unit": unit, "state": state, "cgroup": control, "processes_gone": True}
        if time.monotonic() >= deadline:
            raise RuntimeError("owned stage cleanup not confirmed")
        time.sleep(.05)


def quarantine(path, reason):
    path = Path(path)
    protected(path.parent, directory=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        protected(path)
        return
    with os.fdopen(descriptor, "w") as stream:
        json.dump({"schema_version": 1, "quarantined": True, "reason": reason}, stream)
        stream.flush()
        os.fsync(stream.fileno())


class Lease:
    def __init__(self, config):
        self.config = config
        self.fd = -1

    def __enter__(self):
        if Path(self.config["stop_path"]).exists():
            raise RuntimeError("device STOP/quarantine exists")
        self.fd = os.open(self.config["lock_path"], os.O_RDWR | os.O_NOFOLLOW)
        try:
            info = os.fstat(self.fd)
            expected = self.config["lock_identity"]
            if not stat.S_ISREG(info.st_mode) or [info.st_dev, info.st_ino] != expected:
                raise RuntimeError("original device lock inode changed")
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if Path(self.config["stop_path"]).exists():
                raise RuntimeError("device STOP appeared while claiming lease")
            self.identity = {"device": info.st_dev, "inode": info.st_ino}
            return self
        except BaseException:
            os.close(self.fd)
            self.fd = -1
            raise

    def __exit__(self, *_error):
        # Never LOCK_UN a shared open description: the independent guardian
        # may still own a duplicate after controller/broker death.
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


class Guardian:
    def __init__(self, config, lease, unit, directory, command=None):
        self.owns_command = command is not None
        read_fd, self.write_fd = os.pipe()
        ack_read, ack_write = os.pipe()
        self.receipt = directory / "guardian.json"
        argv = [sys.executable, "-I", str(Path(config["install_root"]) / "guardian.py"),
                "--lock-fd", str(lease.fd), "--read-fd", str(read_fd), "--ack-fd", str(ack_write),
                "--unit", unit, "--stop", config["stop_path"], "--receipt", str(self.receipt)]
        if command is not None:
            path = directory / 'trusted-command.json'
            with path.open('x') as stream:
                json.dump(command, stream)
            path.chmod(0o600)
            argv += ['--command', str(path)]
        self.process = subprocess.Popen(argv, env=CLEAN_ENV, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if command is not None else subprocess.DEVNULL,
            stderr=subprocess.PIPE if command is not None else subprocess.DEVNULL, close_fds=True,
            pass_fds=(lease.fd, read_fd, ack_write), start_new_session=True)
        os.close(read_fd)
        os.close(ack_write)
        try:
            ready, _, _ = select.select([ack_read], [], [], 10)
            if not ready or os.read(ack_read, 1) != b"R":
                raise RuntimeError("independent lease guardian not ready")
        except BaseException:
            os.close(self.write_fd)
            self.write_fd = -1
            self.process.wait(timeout=30)
            raise
        finally:
            os.close(ack_read)

    def finish(self):
        if self.write_fd >= 0:
            os.close(self.write_fd)
            self.write_fd = -1
        self.process.wait(timeout=30)
        if (self.process.returncode and not self.owns_command) or not self.receipt.exists():
            raise RuntimeError("lease guardian cleanup failed")
        value = json.loads(self.receipt.read_text())
        if value.get("cleanup_confirmed") is not True:
            raise RuntimeError("guardian could not confirm cleanup")
        return value


def disconnected(connection):
    if connection is None:
        return False
    ready, _, _ = select.select([connection], [], [], 0)
    return bool(ready) and not connection.recv(1, socket.MSG_PEEK)


class StageExecutor:
    def __init__(self, config, lease):
        if os.geteuid() != 0:
            raise PermissionError("root broker required")
        if lease.fd < 0:
            raise ValueError("active original device lease required")
        self.config, self.lease = config, lease

    def prepare(self):
        directory = Path(self.config["state_root"]) / "stages" / uuid.uuid4().hex
        directory.mkdir(mode=0o711)
        scratch = directory / "scratch"
        scratch.mkdir(mode=0o700)
        os.chown(scratch, self.config["worker_uid"], self.config["worker_gid"])
        return directory, scratch

    def run(self, stage, directory, connection=None):
        # The native negative gates use exactly the production supervisor and
        # start barrier, not a separate weaker synchronous execution path.
        from stages import RunningStage
        protected(directory, directory=True)
        if directory.parent != Path(self.config['state_root']) / 'stages' or stage.scratch != directory / 'scratch':
            raise ValueError('stage is outside root-created probe workspace')
        running = RunningStage(self.config, self.lease, stage, directory, directory.name)
        interrupted = False
        try:
            while not running.done.wait(.05):
                if disconnected(connection):
                    interrupted = True
                    running.stop_event.set()
            receipt = running.receipt
        finally:
            running.stop()
        if interrupted or (running.error and 'InterruptedError' in running.error):
            raise InterruptedError(running.error or 'controller disconnected')
        if running.error:
            raise RuntimeError(running.error)
        return receipt
