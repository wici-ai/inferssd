"""Root-owned configuration and fixed Linux worker namespace construction."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import stat

UNIT = re.compile(r"inferssd-stage-[0-9a-f]{32}\.service\Z")
LIMIT = 1024 * 1024
GATES = (
    "different_execution_uid", "controller_home_denied", "runner_credentials_denied",
    "control_write_denied", "evidence_write_denied", "sudo_denied", "host_sockets_denied",
    "external_network_denied", "user_service_escape_denied", "process_signal_denied",
    "readonly_model_inputs", "cleanup_and_reset_verified", "controller_disconnect_cleanup",
    "original_lock_inode", "stop_quarantine",
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _digest(stream)


def _digest(stream):
    value = hashlib.sha256()
    for data in iter(lambda: stream.read(1024 * 1024), b""):
        value.update(data)
    return value.hexdigest()


def protected(path, *, directory=False):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("absolute canonical root-owned path required")
    for item in (path, *path.parents):
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
            raise ValueError("unsafe owner, symlink or writable path ancestor")
    kind = path.stat().st_mode
    if not (stat.S_ISDIR(kind) if directory else stat.S_ISREG(kind)):
        raise ValueError("unexpected protected path type")
    return path


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("integer out of range")
    return value


def load(path):
    path = protected(path)
    if path.stat().st_size > LIMIT:
        raise ValueError("oversized configuration")
    value = json.loads(path.read_text())
    if value.get("schema_version") != 1:
        raise ValueError("unsupported sandbox configuration")
    for key in ("controller_uid", "controller_gid", "worker_uid", "worker_gid", "client_uid", "client_gid"):
        integer(value[key], 1, 2**31 - 1)
    if len({value[key] for key in ('controller_uid', 'worker_uid', 'client_uid')}) != 3 or len({value[key] for key in ('controller_gid', 'worker_gid', 'client_gid')}) != 3:
        raise ValueError("runner and worker must have distinct identities and groups")
    for key in ("install_root", "state_root"):
        protected(value[key], directory=True)
    protected(Path(value["lock_path"]).parent, directory=True)
    protected(Path(value["stop_path"]).parent, directory=True)
    protected(value["lock_path"])
    for item in value["readonly_system"]:
        protected(item, directory=Path(item).is_dir())
        if Path(item).parts[1] not in ("usr", "lib", "lib64", "bin", "sbin", "opt"):
            raise ValueError("system mounts must be explicit software roots")
        if Path(item) in (Path("/opt"), Path("/usr"), Path("/usr/local")):
            raise ValueError("broad local software mounts need a narrower reviewed root")
    if not value["readonly_system"]:
        raise ValueError("no reviewed toolchain roots")
    for item in value.get("gpu_devices", []):
        path = Path(item)
        if not re.fullmatch(r"/dev/nvidia(?:[0-9]+|ctl|-uvm|-uvm-tools)", item):
            raise ValueError("unreviewed device path")
        if path.is_symlink() or not stat.S_ISCHR(path.stat().st_mode):
            raise ValueError("GPU node is not a real character device")
    integer(value["memory_max"], 128 * 1024**2, 2**50)
    integer(value["tasks_max"], 8, 8192)
    integer(value["log_max"], 4096, 256 * 1024**2)
    integer(value.get('build_max_bytes', 32 * 1024**3), 1024**2, 2**40)
    return value


@dataclass(frozen=True)
class Stage:
    unit: str
    argv: tuple
    readonly: tuple
    scratch: Path
    cwd: str = "/scratch"
    timeout: int = 60
    gpu: bool = False
    environment: tuple = ()


def bwrap(config, stage):
    if not UNIT.fullmatch(stage.unit):
        raise ValueError("not an owned stage unit")
    if not stage.argv or len(stage.argv) > 256 or any(not isinstance(v, str) or "\0" in v for v in stage.argv):
        raise ValueError("invalid worker argv")
    integer(stage.timeout, 1, 43200)
    argv = ["/usr/bin/bwrap", "--unshare-user", "--unshare-pid", "--unshare-ipc",
            "--unshare-uts", "--uid", str(config["worker_uid"]),
            "--gid", str(config["worker_gid"]), "--cap-drop", "ALL", "--new-session",
            "--die-with-parent", "--clearenv", "--hostname", "inferssd-worker",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--tmpfs", "/run",
            "--dir", "/home", "--dir", "/root", "--dir", "/etc", "--dir", "/sys"]
    # systemd supplies a private network namespace; bwrap deliberately retains
    # it so a privileged, trusted measurement client can later join this one.
    for source in config["readonly_system"]:
        argv += ["--ro-bind", source, source]
    for target, source in (("/bin", "usr/bin"), ("/sbin", "usr/sbin"), ("/lib", "usr/lib"), ("/lib64", "usr/lib64")):
        if target not in config["readonly_system"] and Path("/" + source).exists():
            argv += ["--symlink", source, target]
    etc = Path(config["install_root"]) / "etc"
    for name in ("passwd", "group", "nsswitch.conf", "hosts", "resolv.conf"):
        argv += ["--ro-bind", str(etc / name), "/etc/" + name]
    if Path("/etc/ld.so.cache").exists():
        argv += ["--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache"]
    # Keep the host cgroup path paired with this read-only hierarchy. A new
    # cgroup namespace reports '/' while this bind still exposes the host root,
    # hiding memory.max from the engine's adaptive host-memory accounting.
    # ProtectControlGroups plus this read-only bind forbid controller writes.
    # /run and all host sockets remain absent.
    argv += ["--ro-bind", "/sys", "/sys"]
    targets = set()
    for source, target in stage.readonly:
        source = protected(source, directory=Path(source).is_dir())
        if not re.fullmatch(r"/(?:source|models|inputs|probe|build)(?:/[A-Za-z0-9_.-]+)*", target) or target in targets:
            raise ValueError("invalid or overlapping readonly input target")
        if any(target.startswith(old + "/") or old.startswith(target + "/") for old in targets):
            raise ValueError("overlapping input mounts")
        targets.add(target)
        argv += ["--ro-bind", str(source), target]
    argv += ["--bind", str(stage.scratch), "/scratch"]
    if not re.fullmatch(r"/(?:scratch|build)(?:/[A-Za-z0-9_.-]+)*", stage.cwd) or ".." in Path(stage.cwd).parts:
        raise ValueError("worker cwd must be a declared scratch or frozen build path")
    if stage.gpu:
        for path in config["gpu_devices"]:
            argv += ["--dev-bind", path, path]
    argv += ["--setenv", "PATH", "/usr/bin:/bin", "--setenv", "HOME", "/scratch/home",
             "--setenv", "TMPDIR", "/tmp", "--setenv", "LANG", "C.UTF-8",
             "--setenv", "CUDA_VISIBLE_DEVICES", config.get("gpu_uuid", "")]
    for name, value in stage.environment:
        if not re.fullmatch(r"[A-Z_][A-Z_0-9]{0,127}", name) or not isinstance(value, str) or "\0" in value or len(value) > 8192:
            raise ValueError("invalid isolated environment entry")
        argv += ["--setenv", name, value]
    argv += ["--chdir", stage.cwd, "--", *stage.argv]
    return argv


def systemd(config, stage):
    argv = ["/usr/bin/systemd-run", "--quiet", "--wait", "--pipe", "--collect",
            "--service-type=exec", "--unit=" + stage.unit]
    props = ["User=" + str(config["worker_uid"]), "Group=" + str(config["worker_gid"]),
             "SupplementaryGroups=", "PrivateUsers=yes", "PrivateNetwork=yes", "PrivateTmp=yes",
             "ProtectSystem=strict", "ProtectHome=yes", "NoNewPrivileges=yes",
             "RestrictSUIDSGID=yes", "LockPersonality=yes", "KeyringMode=private", "UMask=0077",
             "ProtectKernelTunables=yes", "ProtectKernelModules=yes", "ProtectKernelLogs=yes",
             "ProtectControlGroups=yes", "ProtectClock=yes", "RestrictRealtime=yes",
             "KillMode=control-group", "SendSIGKILL=yes", "TimeoutStopSec=5s", "OOMPolicy=kill",
             "MemoryMax=" + str(config["memory_max"]), "MemorySwapMax=0",
             "LimitMEMLOCK=" + str(config["memory_max"]),
             "TasksMax=" + str(config["tasks_max"]), "RuntimeMaxSec=" + str(stage.timeout),
             "ReadWritePaths=" + str(stage.scratch), "DevicePolicy=closed"]
    for device in config.get("gpu_devices", []) if stage.gpu else []:
        props.append("DeviceAllow=" + device + " rw")
    for item in props:
        argv += ["--property=" + item]
    return argv + ["--", *bwrap(config, stage)]
