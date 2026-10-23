#!/usr/bin/python3
"""Root-owned Mac stage executor. One authenticated connection owns one device lease.

The candidate is always a different UID inside Seatbelt. This module never loads
candidate Python, configuration, or results into the privileged interpreter.
"""
import base64
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import socket
import stat
import subprocess
import time
import uuid

import supervisor as core

IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
JOB_ID = re.compile(r"[0-9a-f]{32}\Z")
ENV_KEYS = frozenset(("LANG", "LC_ALL", "TZ", "CMAKE_BUILD_PARALLEL_LEVEL", "CTEST_PARALLEL_LEVEL"))
MAX_REQUEST = 1024 * 1024


class Shutdown(BaseException):
    pass


def interrupted(number, frame):
    raise Shutdown("broker interrupted")


def require(value, message):
    if not value:
        raise ValueError(message)


def identifier(value):
    require(isinstance(value, str) and IDENTIFIER.fullmatch(value), "invalid job/stage identifier")
    return value


def safe_path(root, value, must_exist=True):
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts, "absolute normalized path required")
    require(path == root or root in path.parents, "path is outside this job")
    # Resolve each component separately so an in-tree symlink is not mistaken for
    # ownership. This is used only when the execution UID has no live processes.
    for item in (path, *path.parents):
        if item == root.parent:
            break
        if item.exists() or item.is_symlink():
            require(not item.is_symlink(), "symlink in controlled path")
        elif must_exist:
            raise ValueError("missing controlled path")
    return path


def peer_identity(peer):
    uid, gid = ctypes.c_uint(), ctypes.c_uint()
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    function = library.getpeereid
    function.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint))
    function.restype = ctypes.c_int
    if function(peer.fileno(), ctypes.byref(uid), ctypes.byref(gid)):
        raise OSError(ctypes.get_errno(), "getpeereid failed")
    return uid.value, gid.value


def birth(pid):
    """Kernel process start time, independent of candidate /proc-like output."""
    # PROC_PIDTBSDINFO=3, struct proc_bsdinfo (Darwin libproc.h).
    class Info(ctypes.Structure):
        _fields_ = [("flags", ctypes.c_uint32), ("status", ctypes.c_uint32),
                    ("xstatus", ctypes.c_uint32), ("pid", ctypes.c_uint32),
                    ("ppid", ctypes.c_uint32), ("uid", ctypes.c_uint32),
                    ("gid", ctypes.c_uint32), ("ruid", ctypes.c_uint32),
                    ("rgid", ctypes.c_uint32), ("svuid", ctypes.c_uint32),
                    ("svgid", ctypes.c_uint32), ("reserved", ctypes.c_uint32),
                    ("comm", ctypes.c_char * 16), ("name", ctypes.c_char * 32),
                    ("nfiles", ctypes.c_uint32), ("pgid", ctypes.c_uint32),
                    ("pjobc", ctypes.c_uint32), ("e_tdev", ctypes.c_uint32),
                    ("e_tpgid", ctypes.c_uint32), ("nice", ctypes.c_int32),
                    ("start_sec", ctypes.c_uint64), ("start_usec", ctypes.c_uint64)]
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    function = library.proc_pidinfo
    function.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int)
    function.restype = ctypes.c_int
    info = Info()
    size = function(pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info))
    if size != ctypes.sizeof(info) or info.pid != pid:
        raise ProcessLookupError(pid)
    return {"pid": pid, "uid": info.uid, "start_sec": info.start_sec, "start_usec": info.start_usec}


def copy_frozen(source, destination, max_bytes):
    """Copy without following links; never chown a candidate inode.

    Caller first proves the dedicated worker UID empty. O_NOFOLLOW and before /
    after inode checks also reject a changed source. Destination must be new and
    beneath a trusted root directory; partial failure stays private for review.
    """
    source = Path(source)
    require(not source.is_symlink() and source.is_dir(), "freeze source must be a directory")
    destination.mkdir(mode=0o700)
    manifest, total = {}, 0
    for directory, names, files in os.walk(source, followlinks=False):
        relative = Path(directory).relative_to(source)
        for name in sorted(names):
            item = Path(directory) / name
            mode = item.lstat().st_mode
            require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), "non-directory in freeze traversal")
            (destination / relative / name).mkdir(mode=0o700)
        for name in sorted(files):
            item = Path(directory) / name
            link = None
            if item.is_symlink():
                resolved = item.resolve(strict=True)
                require(source.resolve() in resolved.parents, "freeze symlink escapes build tree")
                require(resolved.is_file(), "freeze symlink is not a regular file")
                link = {"original_symlink": os.readlink(item), "resolved_relative": str(resolved.relative_to(source.resolve()))}
                item = resolved
            fd = os.open(item, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            try:
                before = os.fstat(fd)
                require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "freeze rejects special files or hardlinks")
                total += before.st_size
                require(total <= max_bytes, "freeze storage budget exceeded")
                target = destination / relative / name
                target_fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                digest = hashlib.sha256()
                with os.fdopen(target_fd, "wb") as output, os.fdopen(os.dup(fd), "rb") as inp:
                    while True:
                        block = inp.read(1024 * 1024)
                        if not block:
                            break
                        digest.update(block)
                        output.write(block)
                after = os.fstat(fd)
                require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "source changed during freeze")
                mode = 0o555 if before.st_mode & 0o111 else 0o444
                target.chmod(mode)
                manifest[str(relative / name)] = {"size": before.st_size, "sha256": digest.hexdigest(), "mode": mode, **(link or {})}
            finally:
                os.close(fd)
    for directory, _, _ in os.walk(destination, topdown=False):
        Path(directory).chmod(0o555)
    return manifest


def protected_tree(path, allowed_parent):
    path = Path(path)
    require(path.is_absolute() and allowed_parent in path.parents, "input outside root staging area")
    for parent in (path, *path.parents):
        info = parent.lstat()
        require(not stat.S_ISLNK(info.st_mode) and info.st_uid == 0 and not info.st_mode & 0o022,
                "input ancestor is not root protected")
    for directory, names, files in os.walk(path, followlinks=False):
        for name in names + files:
            info = (Path(directory) / name).lstat()
            require(info.st_uid == 0 and not info.st_mode & 0o222 and
                    (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)),
                    "staged input is mutable or is a special file")
    return str(path)


def job_manifest(config, job_id):
    path = core.BASE / "private" / "job-manifests" / (job_id + ".json")
    for parent in (path, *path.parents):
        info = parent.lstat()
        require(not stat.S_ISLNK(info.st_mode) and info.st_uid == 0 and not info.st_mode & 0o022,
                "job manifest is not root protected")
    data = json.loads(path.read_text())
    require(data.get("schema_version") == 1 and data.get("job_id") == job_id,
            "job manifest identity mismatch")
    require(isinstance(data.get("input_roots"), list) and data["input_roots"], "job inputs missing")
    require(isinstance(data.get("model_roots"), list), "model roots missing")
    data["input_roots"] = [protected_tree(x, core.BASE / "inputs") for x in data["input_roots"]]
    data["model_roots"] = [protected_tree(x, core.BASE / "models") for x in data["model_roots"]]
    require(len(data["input_roots"]) <= 8 and len(data["model_roots"]) <= 8, "too many input roots")
    data["manifest_sha256"] = core.sha(path)
    return data


def request_stage(request, job, config, admitted_environment=None):
    require(request.get("schema_version") == 1, "unsupported stage schema")
    require(identifier(request.get("job_id")) == job.name, "wrong job")
    identifier(request.get("stage_id"))
    require(request.get("kind") in ("build", "test", "server", "probe"), "unknown stage kind")
    argv = request.get("argv")
    require(isinstance(argv, list) and 0 < len(argv) <= 256 and all(isinstance(x, str) and "\0" not in x and len(x) <= 16384 for x in argv), "invalid argv")
    require(Path(argv[0]).is_absolute(), "stage executable must be absolute")
    environment = request.get("environment", {})
    admitted_environment = admitted_environment or {}
    require(isinstance(environment, dict) and all(isinstance(k, str) and isinstance(v, str) and "\0" not in v for k, v in environment.items()), "invalid environment")
    for key, value in environment.items():
        require(not key.startswith(("DYLD_", "LD_", "PYTHON")) and key not in ("HOME", "PATH", "TMPDIR", "BASH_ENV", "ENV", "ZDOTDIR", "SHELLOPTS", "BASHOPTS"), "loader or process environment forbidden")
        require(key in ENV_KEYS or admitted_environment.get(key) == value, "unapproved inherited environment")
    timeout = request.get("timeout_s")
    require(type(timeout) in (int, float) and 0 < timeout <= config["executor"]["max_stage_seconds"], "invalid stage deadline")
    cwd = safe_path(job, request["cwd"])
    require(cwd.is_dir(), "stage cwd is not a directory")
    return argv, environment, cwd, timeout


class Session:
    def __init__(self, config, peer, job_id, broker_lifeline=None):
        self.config, self.peer = config, peer
        self.broker_lifeline = broker_lifeline
        self.lock_fd = None
        self.stage = None
        self.records = {}
        self.closed = False
        require(isinstance(job_id, str) and JOB_ID.fullmatch(job_id), "job_id must be 32 lowercase hex characters")
        self.job = core.BASE / "jobs" / job_id
        self.manifest = job_manifest(config, job_id)
        require(not self.job.exists(), "job already exists; no replay")
        require(not Path(config["stop"]).exists(), "STOP is present")
        self.lock_fd = core.lease(config)
        try:
            require(not core.members(config["worker_uid"]), "worker UID is not empty")
            self.job.mkdir(mode=0o711)
            for name, mode in (("work", 0o700), ("input", 0o755), ("frozen", 0o755)):
                path = self.job / name
                path.mkdir(mode=mode)
                if name == "work":
                    os.chown(path, config["worker_uid"], config["worker_gid"])
            self.evidence = core.BASE / "evidence" / job_id
            self.evidence.mkdir(mode=0o700)
            self.started = time.monotonic()
            for name in ("worker.sb", "server.sb"):
                profile = (core.BASE / "public" / name).read_text()
                for root in self.manifest["input_roots"] + self.manifest["model_roots"]:
                    profile += "\n(allow file-read* (subpath " + json.dumps(root) + "))\n"
                # Shell canonicalization traverses these directories before entering
                # WORK. Metadata alone reveals no host file contents or directory listing.
                ancestors = {parent for root in [str(self.job / "work"), *self.manifest["input_roots"], *self.manifest["model_roots"]]
                             for parent in Path(root).parents}
                profile += "\n(allow file-read-metadata " + " ".join(
                    "(literal " + json.dumps(str(parent)) + ")" for parent in sorted(ancestors)) + ")\n"
                (self.job / name).write_text(profile)
                (self.job / name).chmod(0o444)
        except BaseException:
            os.close(self.lock_fd)
            self.lock_fd = None
            raise

    def identity(self):
        return {"schema_version": 1, "job_id": self.job.name, "job_root": str(self.job),
                "worker_uid": self.config["worker_uid"], "scope_identity": str(self.job),
                "guardian_pid": os.getpid(),
                "lock_dev_inode": self.config["lock_dev_inode"],
                "manifest_sha256": self.manifest["manifest_sha256"],
                "input_roots": self.manifest["input_roots"], "model_roots": self.manifest["model_roots"]}

    def check_handle(self, request):
        record = self.records.get(request.get("handle"))
        require(record is not None, "unknown stage handle")
        require(request.get("job_id") == self.job.name and request.get("stage_id") == record["stage_id"], "handle belongs to another job/stage")
        return record

    def launch(self, request):
        require(self.stage is None, "only one stage may use the device")
        require(not core.members(self.config["worker_uid"]), "execution UID retained processes")
        admitted = self.manifest.get("stage_environments", {}).get(request.get("stage_id"), {})
        require(isinstance(admitted, dict), "invalid root manifest stage environment")
        argv, extra, cwd, timeout = request_stage(request, self.job, self.config, admitted)
        if argv[0] == "/usr/bin/python3":
            argv = [self.config["tools"]["python"], *argv[1:]]
        require(request["stage_id"] not in {r["stage_id"] for r in self.records.values()}, "stage id replay")
        for name in ("home", "tmp"):
            path = self.job / "work" / name
            if not path.exists():
                path.mkdir(mode=0o700)
                os.chown(path, self.config["worker_uid"], self.config["worker_gid"])
        profile = self.job / ("server.sb" if request["kind"] == "server" else "worker.sb")
        command = ["/usr/bin/sandbox-exec", "-f", str(profile)]
        parameters = {"CLT": self.config["tools"]["clt"], "INPUT": str(self.job / "input"),
                      "PUBLIC": str(core.BASE / "public"), "WORK": str(self.job / "work"),
                      "FROZEN": str(self.job / "frozen"), "MODELS": str(self.job / "input")}
        for key, value in parameters.items():
            command.extend(("-D", key + "=" + value))
        env = {"PATH": "/usr/bin:/bin", "HOME": str(self.job / "work/home"),
               "TMPDIR": str(self.job / "work/tmp"), "LANG": "C", "LC_ALL": "C",
               "PYTHONDONTWRITEBYTECODE": "1", **extra}
        handle = uuid.uuid4().hex
        record = {**self.identity(), "handle": handle, "stage_id": request["stage_id"],
                  "kind": request["kind"], "state": "running", "running": True, "returncode": None, "cleanup_confirmed": False,
                  "started_monotonic": time.monotonic(), "timeout_s": timeout, "argv": argv}
        output = self.evidence / request["stage_id"]
        output.mkdir(mode=0o700)
        def drop():
            core.drop(self.config["worker_uid"], self.config["worker_gid"], str(cwd),
                      {"files": 1024, "file_bytes": 2 << 30, "cpu_seconds": math.ceil(timeout) + 5,
                       "processes": self.config["executor"]["max_processes"]})
        logs = {name: (output / (name + ".log")).open("wb") for name in ("stdout", "stderr")}
        try:
            process = subprocess.Popen(command + argv, env=env, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True, preexec_fn=drop)
        except BaseException:
            for stream in logs.values():
                stream.close()
            raise
        record.update(worker_pid=process.pid, birth_identity=birth(process.pid), evidence=str(output))
        require(record["birth_identity"]["uid"] == self.config["worker_uid"], "worker UID differs")
        self.records[handle] = record
        self.stage = (process, record, logs, {})
        return dict(record)

    def tick(self):
        if self.stage is None:
            return
        process, record, logs, counts = self.stage
        reason = None
        try:
            core.drain_logs(process, logs, counts)
        except RuntimeError:
            reason = "worker log byte bound exceeded"
        if Path(self.config["stop"]).exists():
            reason = "STOP"
        elif core.disconnected(self.peer) or core.disconnected(self.broker_lifeline):
            reason = "controller or broker disconnected"
        elif time.monotonic() - record["started_monotonic"] > record["timeout_s"]:
            reason = "stage deadline"
        else:
            active = core.members(self.config["worker_uid"])
            rss = sum(p["rss_bytes"] for p in active)
            policy = self.config["executor"]
            if len(active) > policy["max_processes"] or rss > policy["memory_budget_bytes"] or core.free_memory() < policy["host_reserve_bytes"]:
                reason = "process or memory reserve budget"
        if process.poll() is not None or reason:
            self.finish(reason)

    def finish(self, reason=None):
        if self.stage is None:
            return
        process, record, logs, counts = self.stage
        returncode = process.poll()
        # Reap the direct child during UID inventory so zombies are not
        # mistaken for an unkillable active worker. Signal actions stay recorded.
        cleanup = core.cleanup(self.config["worker_uid"], reap=process.poll)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cleanup["confirmed"] = False
        try:
            core.drain_logs(process, logs, counts)
        except RuntimeError:
            reason = "worker log byte bound exceeded"
        finally:
            for stream in (*logs.values(), process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()
        record.update(running=False, state="exited" if reason is None and returncode == 0 else "failed",
                      returncode=process.returncode, reason=reason, cleanup=cleanup,
                      cleanup_confirmed=cleanup["confirmed"], elapsed_s=time.monotonic() - record["started_monotonic"])
        if not cleanup["confirmed"]:
            Path(self.config["stop"]).touch(exist_ok=True)
            record["state"] = "quarantined"
        core.save(Path(record["evidence"]) / "result.json", record)
        self.stage = None
        if not cleanup["confirmed"]:
            # Keep this process and its lease alive until empty; the failed result
            # is immutable. No subsequent successful cleanup promotes this run.
            while not core.cleanup(self.config["worker_uid"])["confirmed"]:
                time.sleep(1)
            raise RuntimeError("quarantine cleaned, failed job cannot continue")

    def probe(self, record):
        result = dict(record)
        if record["state"] == "running":
            identity = birth(record["worker_pid"])
            require(identity == record["birth_identity"], "worker PID reused")
            result["members"] = core.members(self.config["worker_uid"])
            result["rss_bytes"] = sum(p["rss_bytes"] for p in result["members"])
            # lsof is an installed Apple tool executed outside the candidate UID.
            observed = subprocess.run(["/usr/sbin/lsof", "-nP", "-a", "-u", str(self.config["worker_uid"]), "-F", "pcuftn"], capture_output=True, text=True, check=False)
            require(observed.returncode in (0, 1), "cannot inventory actual worker files")
            result["lsof"] = observed.stdout
        return result

    def client(self, record):
        require(record["kind"] == "server" and record["state"] == "running", "client requires a running server")
        probe = self.probe(record)
        port = self.config["executor"]["server_port"]
        listeners = subprocess.run(["/usr/sbin/lsof", "-nP", "-a", "-u", str(self.config["worker_uid"]), "-iTCP:" + str(port), "-sTCP:LISTEN", "-F", "pn"], capture_output=True, text=True, check=False)
        require(listeners.returncode == 0, "owned server has no listener")
        names = [line[1:] for line in listeners.stdout.splitlines() if line.startswith("n")]
        pids = [int(line[1:]) for line in listeners.stdout.splitlines() if line.startswith("p")]
        require(names and all(name == "127.0.0.1:" + str(port) for name in names), "listener is not fixed IPv4 loopback")
        require(set(pids) <= {p["pid"] for p in probe["members"]}, "listener outside owned worker UID")
        return {**{k: record[k] for k in ("handle", "worker_pid", "worker_uid", "birth_identity", "scope_identity")},
                "url": "http://127.0.0.1:" + str(port), "listener_pids": pids}

    def dispatch(self, request):
        require(request.get("schema_version") == 1 and request.get("job_id") == self.job.name, "request job/schema mismatch")
        self.tick()
        operation = request.get("operation")
        if operation == "stage.start":
            return self.launch(request)
        if operation == "job.freeze":
            build = self.check_handle(request)
            require(build["kind"] == "build" and build["state"] == "exited" and build["returncode"] == 0 and build["cleanup_confirmed"], "freeze requires the completed successful build handle")
            require(self.stage is None and not core.members(self.config["worker_uid"]), "freeze requires empty execution UID")
            source = safe_path(self.job / "work", request["source"])
            name = identifier(request["name"])
            destination = self.job / "frozen" / name
            manifest = copy_frozen(source, destination, self.config["executor"]["max_artifact_bytes"])
            manifest_path = self.job / "frozen" / (name + ".manifest.json")
            core.save(manifest_path, manifest)
            manifest_path.chmod(0o444)
            return {"path": str(destination), "manifest_path": str(manifest_path),
                    "manifest_sha256": core.sha(manifest_path), "file_count": len(manifest), "readonly": True}
        record = self.check_handle(request)
        if operation == "stage.stop":
            if self.stage and self.stage[1] is record:
                self.finish("requested stop")
            return dict(record)
        if operation == "stage.poll":
            return dict(record)
        if operation == "stage.output":
            name, offset, limit = request.get("stream"), request.get("offset", 0), request.get("limit", 65536)
            require(name in ("stdout", "stderr"), "invalid output stream")
            require(type(offset) is int and offset >= 0 and type(limit) is int and 0 < limit <= 65536, "invalid output range")
            with (Path(record["evidence"]) / (name + ".log")).open("rb") as source:
                size = os.fstat(source.fileno()).st_size
                require(offset <= size, "output offset exceeds log")
                source.seek(offset)
                chunk = source.read(limit)
            return {"offset": offset, "next_offset": offset + len(chunk),
                    "data_b64": base64.b64encode(chunk).decode("ascii"),
                    "eof": not record["running"] and offset + len(chunk) == size}
        if operation == "stage.wait":
            wait = request.get("wait_s", 1)
            require(type(wait) in (int, float) and 0 <= wait <= 30, "invalid bounded wait")
            until = time.monotonic() + wait
            while record["state"] == "running" and time.monotonic() < until:
                self.tick()
                time.sleep(.1)
            return dict(record)
        if operation == "stage.probe":
            return self.probe(record)
        if operation == "stage.client":
            return self.client(record)
        raise ValueError("unknown operation")

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.finish("controller disconnected or job finished")
            require(not core.members(self.config["worker_uid"]), "worker not empty at lease release")
        finally:
            # Never unlock when an unaccounted process still uses the device.
            if core.members(self.config["worker_uid"]):
                Path(self.config["stop"]).touch(exist_ok=True)
                while not core.cleanup(self.config["worker_uid"])["confirmed"]:
                    time.sleep(1)
            if self.lock_fd is not None:
                os.close(self.lock_fd)
                self.lock_fd = None


def serve_connection(config, peer, broker_lifeline=None):
    require(peer_identity(peer) == (config["runner_uid"], config["runner_gid"]), "unauthorized coordinator UID")
    session, pending = None, b""
    try:
        while True:
            if core.disconnected(broker_lifeline):
                raise Shutdown("root broker disconnected")
            if session:
                session.tick()
            ready, _, _ = select.select([peer], [], [], .1)
            if not ready:
                continue
            block = peer.recv(65536)
            if not block:
                break
            pending += block
            require(len(pending) <= MAX_REQUEST, "oversized stage request")
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                request = json.loads(line)
                if session is None:
                    require(request.get("operation") == "job.open" and request.get("schema_version") == 1, "job.open required")
                    session = Session(config, peer, request["job_id"], broker_lifeline)
                    result = session.identity()
                elif request.get("operation") == "job.finish":
                    require(request.get("job_id") == session.job.name, "wrong close job")
                    session.close()
                    peer.sendall(json.dumps({"status": "closed", "cleanup_confirmed": True}).encode() + b"\n")
                    return
                else:
                    result = session.dispatch(request)
                peer.sendall(json.dumps(result, sort_keys=True).encode() + b"\n")
    finally:
        if session:
            session.close()


def serve(config):
    require(config.get("executor", {}).get("enabled") is True, "stage executor remains disabled pending native proof")
    path = core.BASE / "control" / "executor.sock"
    require(not path.exists(), "broker socket already exists; inspect its owner")
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, interrupted)
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(str(path))
    os.chown(path, 0, config["runner_gid"])
    path.chmod(0o660)
    listener.listen(1)
    try:
        while True:
            peer, _ = listener.accept()
            # A separate guardian owns the peer, device lease, and worker. The
            # root broker keeps only the other lifeline endpoint. SIGKILL of the
            # broker closes that endpoint; the guardian cleans before unlocking.
            parent_life, child_life = socket.socketpair()
            pid = os.fork()
            if pid == 0:
                parent_life.close()
                listener.close()
                os.setsid()
                code = 0
                try:
                    serve_connection(config, peer, child_life)
                except (ValueError, RuntimeError, OSError) as error:
                    code = 1
                    try:
                        peer.sendall(json.dumps({"status": "failed", "error": str(error)}).encode() + b"\n")
                    except OSError:
                        pass
                except BaseException:
                    code = 1
                finally:
                    peer.close()
                    child_life.close()
                os._exit(code)
            child_life.close()
            peer.close()
            try:
                os.waitpid(pid, 0)
            finally:
                parent_life.close()
    finally:
        listener.close()
        path.unlink()


def main():
    serve(core.config())


if __name__ == "__main__":
    main()
