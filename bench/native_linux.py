"""Linux finite scope, original flock lease and actual ELF/source receipts."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

import native_common as common
from native_lease import Lease, LeaseServer

HERE = Path(__file__).resolve().parent


def scope_pids(root):
    root = Path(root)
    # Unknown/disappeared membership is never evidence of a clean live scope.
    found = {int(x) for x in (root / "cgroup.procs").read_text().split()}
    for path in root.rglob("cgroup.procs"):
        found.update(int(x) for x in path.read_text().split())
    return found


def cgroup_snapshot(pid, budget, mount=Path("/sys/fs/cgroup"), proc=Path("/proc")):
    raw = (proc / str(pid) / "cgroup").read_text()
    paths = [line[3:] for line in raw.splitlines() if line.startswith("0::")]
    if len(paths) != 1 or ".." in Path(paths[0]).parts:
        raise ValueError("one visible unified cgroup required")
    current, rows = mount / paths[0].lstrip("/"), []
    while True:
        row = {"path": str(current)}
        for key in ("memory.max", "memory.swap.max", "memory.current", "memory.peak", "memory.events", "memory.stat"):
            path = current / key
            if path.exists():
                row[key] = path.read_text()
        rows.append(row)
        if current == mount:
            break
        current = current.parent
    limits = [int(row["memory.max"]) for row in rows if row.get("memory.max", "max").strip() != "max"]
    swaps = [int(row["memory.swap.max"]) for row in rows if row.get("memory.swap.max", "max").strip() != "max"]
    if (not limits or min(limits) != budget or not swaps or min(swaps) != 0
            or rows[0].get("memory.max", "").strip() != str(budget)
            or rows[0].get("memory.swap.max", "").strip() != "0" or "memory.peak" not in rows[0]):
        raise ValueError("actual scope/ancestor memory limit, zero swap or peak counter differs")
    for row in rows:
        if "memory.max" in row and not all(k in row.get("memory.events", "") for k in ("oom ", "oom_kill ")):
            raise ValueError("missing ancestor OOM counters")
    return {"path": rows[0]["path"], "raw": raw, "ancestors": rows}


def unchanged_oom(before, after):
    if before["raw"] != after["raw"] or len(before["ancestors"]) != len(after["ancestors"]):
        raise ValueError("resource scope changed")
    for left, right in zip(before["ancestors"], after["ancestors"]):
        parse = lambda row: dict(line.split() for line in row.get("memory.events", "").splitlines())
        a, b = parse(left), parse(right)
        if left["path"] != right["path"] or any(a.get(k, "0") != b.get(k, "0") for k in ("oom", "oom_kill", "oom_group_kill")):
            raise ValueError("scope or ancestor OOM counter changed")


def mapped_images(pid, profile, out):
    raw = Path(f"/proc/{pid}/maps").read_text()
    Path(str(out) + ".maps").write_text(raw)
    expected = {str(Path(x["path"]).resolve()): x["sha256"] for x in [profile["binary"], *profile["images"]]}
    images = {}
    for line in raw.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6 or not fields[5].startswith("/"):
            continue
        name = Path(fields[5]).name
        if not (name.startswith(("libggml", "libllama", "libmtmd", "libkimi", "liburing"))
                or fields[5] in expected):
            continue
        if fields[5].endswith(" (deleted)"):
            raise ValueError("deleted project mapping")
        actual = common.file_identity(fields[5], True)
        major, minor = (int(x, 16) for x in fields[3].split(":"))
        if actual["dev"] != os.makedev(major, minor) or actual["ino"] != int(fields[4]):
            raise ValueError("mapped image inode differs from hashed file")
        if expected.get(actual["path"]) != actual["sha256"]:
            raise ValueError("foreign or changed mapped project image")
        images[actual["path"]] = actual["sha256"]
    executable = common.file_identity(f"/proc/{pid}/exe", True)
    if (executable["path"] != str(Path(profile["binary"]["path"]).resolve())
            or executable["sha256"] != profile["binary"]["sha256"] or set(images) != set(expected)):
        raise ValueError("actual executable or complete mapped-image set differs")
    result = [{"path": path, "sha256": images[path]} for path in sorted(images) if path != executable["path"]]
    common.save(Path(str(out) + ".json"), result)
    return result


class LinuxRuntime:
    def __init__(self, profile, out):
        self.profile, self.out = profile, out
        self.before = None

    def prepare(self):
        if sys.platform != "linux":
            raise ValueError("Linux profile requires Linux")
        text = Path("/proc/self/mountinfo").read_text()
        if not any(" - cgroup2 " in line and line.split()[3:5] == ["/", "/sys/fs/cgroup"] for line in text.splitlines()):
            raise ValueError("full cgroup2 root must be visible")
        self.before = cgroup_snapshot(os.getpid(), self.profile["resources"]["memory_budget_bytes"])
        allowed = sorted(os.sched_getaffinity(0))
        if len(allowed) < self.profile["resources"]["minimum_cpus"]:
            raise ValueError("insufficient allowed CPUs")
        host = {"uname": list(os.uname()), "affinity": allowed, "cpuinfo": Path("/proc/cpuinfo").read_text(),
                "meminfo": Path("/proc/meminfo").read_text(), "mountinfo": text}
        if any(Path(x["path"]).name.startswith("libggml-cuda") for x in self.profile["images"]):
            host["gpu"] = common.command(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"])
        common.save(self.out / "linux-host.json", host)
        common.save(self.out / "cgroup-before.json", self.before)
        self.sample(None)

    def process_identity(self, pid):
        raw = Path(f"/proc/{pid}/stat").read_text()
        start = raw[raw.rfind(")") + 2:].split()[19]
        return json.dumps({"pid": pid, "start_ticks": start,
                           "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                           "exe": str(Path(f"/proc/{pid}/exe").resolve(strict=True))}, sort_keys=True)

    def confirm_child(self, pid, lock_fd):
        child = cgroup_snapshot(pid, self.profile["resources"]["memory_budget_bytes"])
        if child["raw"] != self.before["raw"]:
            raise ValueError("server escaped the finite scope")
        held, inherited = Path(f"/proc/{pid}/fd/{lock_fd}").stat(), os.fstat(lock_fd)
        if (held.st_dev, held.st_ino) != (inherited.st_dev, inherited.st_ino):
            raise ValueError("server did not inherit the device lock")
        raw = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        actual = dict(item.decode().split("=", 1) for item in raw if b"=" in item)
        expected = common.clean_environment(self.profile)
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError("server environment differs from the reviewed launch")
        common.save(self.out / "effective-environment.json", {key: actual[key] for key in expected
                    if key in self.profile["environment"] or key == "LD_LIBRARY_PATH"})

    def confirm_port(self, pid, port):
        sockets = set()
        for path in Path(f"/proc/{pid}/fd").iterdir():
            try:
                target = os.readlink(path)
                if target.startswith("socket:["):
                    sockets.add(target[8:-1])
            except FileNotFoundError:
                continue
        rows = [line.split() for line in Path(f"/proc/{pid}/net/tcp").read_text().splitlines()[1:]]
        if not any(row[1] == f"0100007F:{port:04X}" and row[3] == "0A" and row[9] in sockets for row in rows):
            raise ValueError("health port is not owned by the exact server PID")

    def images(self, pid, name):
        return mapped_images(pid, self.profile, self.out / name)

    def sample(self, pid):
        now = cgroup_snapshot(os.getpid(), self.profile["resources"]["memory_budget_bytes"])
        if self.before is not None:
            unchanged_oom(self.before, now)
        memory = Path("/proc/meminfo").read_text()
        available = int(next(line.split()[1] for line in memory.splitlines() if line.startswith("MemAvailable:"))) * 1024
        row = {"monotonic_s": time.monotonic(), "cgroup": now, "meminfo": memory,
               "host_available_bytes": available, "diskstats": Path("/proc/diskstats").read_text()}
        if pid is not None:
            row["process_status"] = Path(f"/proc/{pid}/status").read_text()
            if cgroup_snapshot(pid, self.profile["resources"]["memory_budget_bytes"])["raw"] != now["raw"]:
                raise ValueError("server moved out of its finite scope")
        with (self.out / "memory-samples.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if available < self.profile["resources"]["host_reserve_bytes"]:
            raise RuntimeError("host available memory fell below the declared reserve")

    def host_receipt(self):
        host = json.loads((self.out / "linux-host.json").read_text())
        stable_cpu = sorted({line for line in host["cpuinfo"].splitlines() if line.split(":", 1)[0].strip() in
                             ("model name", "vendor_id", "Hardware", "CPU implementer", "CPU architecture", "CPU part", "Features", "flags")})
        return {"hardware": json.dumps({"cpu": stable_cpu, "gpu": host.get("gpu"), "affinity": host["affinity"]}, sort_keys=True),
                "os": " ".join(host["uname"]), "memory_budget_bytes": self.profile["resources"]["memory_budget_bytes"],
                "resource_checks_passed": True, "effective_settings": common.effective_settings(self.profile),
                "budget_enforcement": "cgroup v2 hard memory.max, zero swap; GPU allocations are not claimed to be fully charged",
                "os_page_cache_treatment": "none; OS cache state uncontrolled"}

    def finish(self):
        if self.before is None:
            return False
        self.sample(None)
        after = cgroup_snapshot(os.getpid(), self.profile["resources"]["memory_budget_bytes"])
        common.save(self.out / "cgroup-after.json", after)
        remaining = scope_pids(after["path"]) - {os.getpid()}
        if remaining:
            raise RuntimeError(f"other scope processes remain: {sorted(remaining)}")
        return False  # Only the controller can verify this worker's scope is gone.

    def evidence_paths(self):
        return [x for x in self.out.iterdir() if x.name.startswith(("linux-host", "cgroup-", "memory-samples", "images-", "effective-environment"))]


def systemctl(*args):
    result = subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True, timeout=10)
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def scope_state(unit):
    result = systemctl("show", unit, "--property=LoadState", "--property=ActiveState", "--property=ControlGroup")
    fields = dict(line.split("=", 1) for line in result["stdout"].splitlines() if "=" in line)
    group, members = fields.get("ControlGroup", ""), []
    if group:
        root = Path("/sys/fs/cgroup") / group.lstrip("/")
        if root.exists():
            members = sorted(scope_pids(root))
    known = fields.get("LoadState") == "not-found" or (result["returncode"] == 0 and
            fields.get("ActiveState") in ("inactive", "failed") and not members)
    return {"query": result, "fields": fields, "pids": members, "gone": known and not members}


def run_scope(profile, environment, lock_fd):
    if sys.platform != "linux":
        raise ValueError("Linux scope requires Linux")
    out = Path(environment["SSDASH_RUN_DIR"])
    unit = f"ssdash-final-{uuid.uuid4().hex}.scope"
    profile_path = out / "scope-profile.json"
    common.save(profile_path, profile)
    lease = LeaseServer(lock_fd, unit)
    lock_path = str(Path(f"/proc/self/fd/{lock_fd}").resolve(strict=True))
    argv = ["systemd-run", "--user", "--scope", "--quiet", "--unit", unit,
            "-p", "Delegate=yes", "-p", f"MemoryMax={profile['resources']['memory_budget_bytes']}",
            "-p", "MemorySwapMax=0", sys.executable, str(Path(__file__).resolve()), "--worker",
            "--profile", str(profile_path), "--socket", lease.path, "--lock", lock_path]
    common.save(out / "scope-launch.json", {"argv": argv, "unit": unit})
    signals, launcher, failure = [], None, None
    handlers = {s: signal.signal(s, lambda n, _f: signals.append(n)) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    deadline = time.monotonic() + profile["load_timeout_s"] + float(environment["SSDASH_REQUEST_TIMEOUT"]) + 60
    try:
        with (out / "scope.log").open("xb") as log:
            launcher = subprocess.Popen(argv, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while launcher.poll() is None:
                lease.poll()
                if signals or time.monotonic() > deadline:
                    raise InterruptedError("native scope interrupted or timed out")
                time.sleep(.05)
        if launcher.returncode != 0 or lease.peer is None:
            raise RuntimeError("scope worker did not complete successfully with the inherited lock")
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        # The exact owned scope is stopped before either controller lease closes.
        cleanup = {"unit": unit, "worker": lease.peer, "actions": []}
        while True:
            try:
                cleanup["actions"].append(systemctl("stop", unit))
                if launcher is not None and common.group_exists(launcher):
                    cleanup["launcher"] = common.stop_owned(launcher, 5)
                cleanup["state"] = scope_state(unit)
                clean = cleanup["state"]["gone"] and (launcher is None or not common.group_exists(launcher))
            except BaseException as error:
                cleanup["error"] = str(error)
                clean = False
            common.save(out / "scope-cleanup.json", cleanup)
            if clean:
                break
            failure = failure or "scope cleanup uncertain; retaining inherited device lock"
            common.save(out / "quarantine.json", {"unit": unit, "controller_pid": os.getpid(), "reason": failure})
            time.sleep(1)
        path = Path(environment["SSDASH_WRAPPER_RECEIPT"])
        receipt = json.loads(path.read_text()) if path.exists() else {"status": "failed", "server": {}, "evidence": []}
        receipt["server"]["resource_scope_gone"] = True
        if failure or signals:
            receipt.update(status="failed", scope_error=failure, scope_signals=signals)
        for item in (out / "scope-launch.json", out / "scope-cleanup.json", out / "scope.log"):
            if item.exists():
                receipt["evidence"].append({"path": item.name, "sha256": common.file_identity(item, True)["sha256"]})
        common.save(path, receipt)
        lease.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return 0 if receipt["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    lease = Lease(args.socket, args.lock)
    runtime = None
    try:
        import native_wrapper
        profile = common.validate_profile(json.loads(args.profile.read_text()))
        environment = dict(os.environ, SSDASH_LOCK_FD=str(lease.fd))
        runtime = LinuxRuntime(profile, Path(environment["SSDASH_RUN_DIR"]))
        return native_wrapper.run_native(profile, environment, runtime, lease.fd, lease)
    finally:
        # Controller SIGKILL/disconnect must not release the last lease while
        # another process remains in the finite scope (including an escaped group).
        if runtime is not None and runtime.before is not None:
            while True:
                try:
                    if scope_pids(runtime.before["path"]) == {os.getpid()}:
                        break
                except OSError:
                    pass
                common.save(runtime.out / "worker-quarantine.json", {"pid": os.getpid(), "retaining_lock": True})
                time.sleep(1)
        lease.close()


if __name__ == "__main__":
    raise SystemExit(main())
