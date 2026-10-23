"""Darwin owned process and sampled soft-budget receipts; no cgroup claim."""
import json
import argparse
import os
from pathlib import Path
import re
import sys
import time

import native_common as common
from macos.measurement_images import snapshot_images
from native_lease import Lease
from native_core import CORE


def parse_vm(text):
    page = re.search(r"page size of (\d+) bytes", text)
    if page is None:
        raise ValueError("vm_stat page size missing")
    counts = {name.strip(): int(count) for name, count in re.findall(r"^([^:\n]+):\s*(\d+)\.", text, re.M)}
    for name in ("Pages free", "Pages inactive", "Pages occupied by compressor", "Pageouts"):
        if name not in counts:
            raise ValueError(f"vm_stat counter missing: {name}")
    return {"page_size": int(page.group(1)), "pages": counts}


def parse_swap(text):
    match = re.search(r"used\s*=\s*([0-9.]+)([KMG])", text)
    if match is None:
        raise ValueError("vm.swapusage used bytes missing")
    return int(float(match.group(1)) * {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}[match.group(2)])


def enforce_sample(sample, baseline, limits):
    if sample["reclaimable_estimate_bytes"] < limits["host_reserve_bytes"]:
        raise RuntimeError("sampled free+inactive memory estimate fell below reserve")
    if sample["rss_bytes"] > limits["memory_budget_bytes"]:
        raise RuntimeError("sampled process RSS exceeded soft budget")
    if sample["compressor_bytes"] - baseline["compressor_bytes"] > limits["max_compressor_growth_bytes"]:
        raise RuntimeError("system compressor growth exceeded the declared soft threshold")
    if sample["swap_used_bytes"] - baseline["swap_used_bytes"] > limits["max_swap_growth_bytes"]:
        raise RuntimeError("system swap growth exceeded the declared soft threshold")


class MacRuntime:
    def __init__(self, profile, out):
        self.profile, self.out = profile, out
        self.baseline = None

    def prepare(self):
        if sys.platform != "darwin":
            raise ValueError("Mac profile requires Darwin")
        hardware = {name: common.command(["sysctl", "-n", name]).strip()
                    for name in ("hw.memsize", "hw.logicalcpu", "machdep.cpu.brand_string", "hw.model")}
        if int(hardware["hw.logicalcpu"]) < self.profile["resources"]["minimum_cpus"]:
            raise ValueError("insufficient logical CPUs")
        if self.profile["resources"]["memory_budget_bytes"] + self.profile["resources"]["host_reserve_bytes"] > int(hardware["hw.memsize"]):
            raise ValueError("soft budget plus reserve exceeds physical memory")
        common.save(self.out / "mac-host.json", {"hardware": hardware, "os": common.command(["sw_vers"]),
                                                "uname": list(os.uname()),
                                                "storage": common.command(["df", "-k", self.profile["shards"][0]["path"]])})
        self.sample(None)

    def process_identity(self, pid):
        value = common.command(["ps", "-o", "uid=,lstart=,comm=", "-p", str(pid)], timeout=2).strip().split(None, 6)
        if len(value) != 7 or int(value[0]) != os.getuid():
            raise ValueError("owned process identity unavailable")
        return json.dumps({"pid": pid, "uid": value[0], "start": " ".join(value[1:6]), "command": value[6]}, sort_keys=True)

    def confirm_child(self, pid, lock_fd):
        # pass_fds preserves the original open-file-description; do not acquire
        # another lock. Record that exact fd's device/inode for the receipt.
        info = os.fstat(lock_fd)
        raw = common.command(["/usr/sbin/lsof", "-nP", "-a", "-p", str(pid), "-d", str(lock_fd), "-F0Dfin"])
        fields = raw.split("\0")
        inodes = [int(x[1:]) for x in fields if x.startswith("i")]
        devices = [int(x[1:], 16) for x in fields if x.startswith("D")]
        if inodes != [info.st_ino] or devices != [info.st_dev & 0xffffffff]:
            raise ValueError("server did not retain the inherited lock file")
        (self.out / "lock-fd.lsof").write_text(raw)

    def confirm_port(self, pid, port):
        raw = common.command(["/usr/sbin/lsof", "-nP", "-a", "-p", str(pid), "-iTCP:" + str(port), "-sTCP:LISTEN", "-F0pn"])
        fields = [x.lstrip("\n") for x in raw.split("\0") if x.lstrip("\n")]
        if f"p{pid}" not in fields or f"n127.0.0.1:{port}" not in fields:
            raise ValueError("health port is not owned by the exact server PID")
        (self.out / "listener.lsof").write_text(raw)

    def images(self, pid, name):
        value = snapshot_images(pid, self.profile["binary"]["path"], self.out / name, require_backends=True)
        expected = {str(Path(x["path"]).resolve()): x["sha256"] for x in [self.profile["binary"], *self.profile["images"]]}
        actual = {x["path"]: x["sha256"] for x in value["images"]}
        if actual != expected:
            raise ValueError("actual complete Mac image set differs from reviewed bundle")
        binary = str(Path(self.profile["binary"]["path"]).resolve())
        return [{"path": path, "sha256": actual[path]} for path in sorted(actual) if path != binary]

    def sample(self, pid):
        begin = time.monotonic()
        raw_vm = common.command(["vm_stat"])
        raw_swap = common.command(["sysctl", "-n", "vm.swapusage"])
        vm = parse_vm(raw_vm)
        size, pages = vm["page_size"], vm["pages"]
        rss = int(common.command(["ps", "-o", "rss=", "-p", str(pid)], timeout=2).strip()) * 1024 if pid else 0
        sample = {"sample_begin": begin, "sample_end": time.monotonic(), "rss_bytes": rss,
                  "reclaimable_estimate_bytes": (pages["Pages free"] + pages["Pages inactive"]) * size,
                  "compressor_bytes": pages["Pages occupied by compressor"] * size,
                  "swap_used_bytes": parse_swap(raw_swap), "raw_vm_stat": raw_vm, "raw_swap": raw_swap,
                  "definition": "non-atomic sampled process RSS and system free+inactive estimate; not a unified-memory hard limit"}
        if self.baseline is None:
            self.baseline = sample
        with (self.out / "memory-samples.jsonl").open("a") as stream:
            stream.write(json.dumps(sample) + "\n")
        enforce_sample(sample, self.baseline, self.profile["resources"])

    def host_receipt(self):
        host = json.loads((self.out / "mac-host.json").read_text())
        return {"hardware": json.dumps(host["hardware"], sort_keys=True), "os": host["os"],
                "memory_budget_bytes": self.profile["resources"]["memory_budget_bytes"],
                "resource_checks_passed": True, "effective_settings": common.effective_settings(self.profile),
                "budget_enforcement": "soft sampled RSS/reserve/compressor/swap thresholds; no OS-enforced per-process memory cap",
                "os_page_cache_treatment": "none; OS cache state uncontrolled"}

    def finish(self):
        if self.baseline is not None:
            self.sample(None)
        return True  # This adapter creates no external native resource scope.

    def evidence_paths(self):
        return [x for x in self.out.iterdir() if x.name.startswith(("mac-host", "memory-samples", "images-", "listener.", "lock-fd."))]


def cleanup_confirmed(path):
    try:
        server = json.loads(path.read_text())["server"]
        return server.get("owned_processes_gone") is True and server.get("resource_scope_gone") is True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def guard_worker(argv, environment, lock_fd, deadline_s):
    return CORE["guardian"].guard_worker(
        argv, environment, lock_fd, deadline_s, effects=common,
        cleanup_confirmed=cleanup_confirmed)

def run_guardian(profile, environment, lock_fd):
    if sys.platform != "darwin":
        raise ValueError("Mac guardian requires Darwin")
    path = Path(environment["SSDASH_RUN_DIR"]) / "guardian-profile.json"
    common.save(path, profile)
    argv = [sys.executable, str(Path(__file__).resolve()), "--worker", "--profile", str(path)]
    return guard_worker(argv, environment, lock_fd,
                        profile["load_timeout_s"] + float(environment["SSDASH_REQUEST_TIMEOUT"]) + 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--lock-dev", type=int, required=True)
    parser.add_argument("--lock-ino", type=int, required=True)
    args = parser.parse_args()
    lease = Lease(args.socket, expected_identity=(args.lock_dev, args.lock_ino))
    environment = dict(os.environ, SSDASH_LOCK_FD=str(lease.fd))
    try:
        import native_wrapper
        profile = common.validate_profile(json.loads(args.profile.read_text()))
        return native_wrapper.run_native(profile, environment, MacRuntime(profile, Path(environment["SSDASH_RUN_DIR"])), lease.fd, lease)
    finally:
        path = Path(environment["SSDASH_WRAPPER_RECEIPT"])
        while not cleanup_confirmed(path):
            common.save(path.parent / "worker-quarantine.json", {"pid": os.getpid(), "retaining_lock": True})
            time.sleep(1)
        lease.close()


if __name__ == "__main__":
    raise SystemExit(main())
