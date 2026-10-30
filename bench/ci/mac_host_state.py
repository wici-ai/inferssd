"""Read-only, bounded Mac host observations from the trusted coordinator.

These sampled checks are not isolation or proof that no short-lived work ran.
Commands never request argv or environment. Policy limits are caller supplied.
"""

from decimal import Decimal
import math
import os
import re
import selectors
import subprocess
import sys
import time


COUNTERS = ("swapins", "swapouts", "compressions", "decompressions", "pageins", "pageouts")
COMMANDS = {
    "vm_stat": ("/usr/bin/vm_stat",),
    "swap": ("/usr/sbin/sysctl", "-n", "vm.swapusage"),
    "memory_bytes": ("/usr/sbin/sysctl", "-n", "hw.memsize"),
    "memory_pressure_level": ("/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"),
    "processes": ("/bin/ps", "-axo", "pid=,ppid=,uid=,time=,etime=,rss=,comm="),
}
LIMITS = ("min_available_bytes", "max_swapout_bytes", "max_compressed_bytes",
          "max_non_ci_cpu_cores", "max_non_ci_process_cpu_cores", "max_non_ci_process_rss_bytes")


def _bounded_command(argv, timeout_s, max_output_bytes):
    """Cap combined pipe bytes while reading, and reap only our own command."""
    raw = {"stdout": bytearray(), "stderr": bytearray()}
    result = dict(returncode=None, error=None)
    started = time.monotonic()
    process = None
    try:
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "LC_ALL": "C", "LANG": "C"})
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            while selector.get_map():
                remaining = timeout_s - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("command deadline exceeded")
                for key, _ in selector.select(remaining):
                    capacity = max_output_bytes - sum(len(value) for value in raw.values())
                    block = os.read(key.fd, min(65536, capacity + 1))
                    raw[key.data].extend(block[:capacity])
                    if len(block) > capacity:
                        raise ValueError("command output byte limit exceeded")
                    if not block:
                        selector.unregister(key.fileobj)
            process.wait(timeout=max(0.001, timeout_s - (time.monotonic() - started)))
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        result["error"] = type(error).__name__ + ": " + str(error)
    finally:
        if process is not None:
            if process.poll() is None:
                try:
                    process.kill()
                    process.wait(timeout=1)
                except (OSError, subprocess.TimeoutExpired) as error:
                    result["error"] = type(error).__name__ + ": command cleanup failed"
            result["returncode"] = process.returncode
            process.stdout.close()
            process.stderr.close()
    result.update({key: bytes(value).decode("utf-8", errors="replace") for key, value in raw.items()})
    result["elapsed_s"] = time.monotonic() - started
    return result


def parse_vm_stat(raw):
    header = re.search(r"page size of (\d+) bytes", raw)
    fields = {key.strip('"').lower().replace(" ", "_").replace("-", "_"): int(value)
              for key, value in re.findall(r"(?m)^([^:\n]+):\s*(\d+)\.\s*$", raw)}
    if not header or int(header[1]) <= 0 or any(key not in fields for key in (*COUNTERS, "pages_free", "file_backed_pages")):
        raise ValueError("incomplete vm_stat counters")
    pages = {key[6:]: value for key, value in fields.items() if key.startswith("pages_")}
    pages.update({key[:-6]: value for key, value in fields.items() if key.endswith("_pages")})
    return dict(page_size_bytes=int(header[1]), pages=pages, counters={key: fields[key] for key in COUNTERS})


def parse_swap(raw):
    fields = {}
    for key, number, unit in re.findall(r"\b(total|used|free)\s*=\s*(\d+(?:\.\d+)?)\s*([BKMGT])\b", raw):
        fields[key + "_bytes"] = int(Decimal(number) * (1024 ** "BKMGT".index(unit)))
    if set(fields) != {"total_bytes", "used_bytes", "free_bytes"}:
        raise ValueError("incomplete swap usage")
    if max(fields["used_bytes"], fields["free_bytes"]) > fields["total_bytes"]:
        raise ValueError("invalid swap usage")
    return fields


def _time_ms(value):
    match = re.fullmatch(r"(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)(?:\.(\d{1,3}))?", value)
    if not match:
        raise ValueError("invalid ps elapsed/CPU time")
    days, hours, minutes, seconds, fraction = match.groups()
    if int(seconds) >= 60 or hours is not None and int(minutes) >= 60:
        raise ValueError("invalid ps time component")
    return ((int(days or 0) * 86400 + int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds)) * 1000
            + int((fraction or "").ljust(3, "0")))


def parse_processes(raw):
    result, pids = [], set()
    for line in raw.splitlines():
        parts = line.split(None, 6)
        if len(parts) != 7:
            raise ValueError("incomplete process inventory")
        pid, ppid, uid = (int(value) for value in parts[:3])
        rss = int(parts[5]) * 1024
        if pid <= 0 or pid in pids or min(ppid, uid, rss) < 0:
            raise ValueError("invalid or duplicate process identity")
        pids.add(pid)
        result.append(dict(pid=pid, ppid=ppid, uid=uid, cpu_time_ms=_time_ms(parts[3]),
            elapsed_ms=_time_ms(parts[4]), rss_bytes=rss, comm=parts[6], name=os.path.basename(parts[6])))
    if not result:
        raise ValueError("empty process inventory")
    return result


def snapshot(*, timeout_s=5, max_output_bytes=2 << 20):
    """Collect five fixed commands once; retain partial evidence on failure."""
    if (type(timeout_s) not in (int, float) or not 0 < timeout_s <= 10
            or type(max_output_bytes) is not int or not 1024 <= max_output_bytes <= 4 << 20):
        raise ValueError("finite command timeout/output bounds required")
    result = dict(schema_version=1, platform=sys.platform, status="passed", observer_pid=os.getpid(),
        started_monotonic_ns=time.monotonic_ns(), raw={}, errors=[], unavailable=[], memory_pressure_level=None)
    if sys.platform != "darwin":
        result["errors"].append(dict(source="platform", reason="Darwin host required"))
    else:
        parsers = dict(vm_stat=parse_vm_stat, swap=parse_swap, processes=parse_processes,
                       memory_bytes=lambda value: int(value.strip()), memory_pressure_level=lambda value: int(value.strip()))
        for key, argv in COMMANDS.items():
            raw = _bounded_command(argv, timeout_s, max_output_bytes)
            result["raw"][key] = raw
            try:
                if raw["error"] or raw["returncode"] != 0:
                    raise ValueError(raw["error"] or "command returned " + str(raw["returncode"]))
                result[key] = parsers[key](raw["stdout"])
                if key in ("memory_bytes", "memory_pressure_level") and result[key] < (1 if key == "memory_bytes" else 0):
                    raise ValueError("invalid sysctl value")
            except (ValueError, TypeError) as error:
                target = "unavailable" if key == "memory_pressure_level" else "errors"
                if key == "memory_pressure_level":
                    result[key] = None
                result[target].append(dict(source=key, reason=str(error)))
    result["finished_monotonic_ns"] = time.monotonic_ns()
    result["status"] = "failed" if result["errors"] else "passed"
    return result


def _policy(policy):
    keys = {*LIMITS, "ci_uids", "ignore_pids", "normal_pressure_levels", "require_pressure_level"}
    if not isinstance(policy, dict) or set(policy) - keys:
        raise ValueError("unknown host-state policy field")
    for key in LIMITS:
        if key in policy and (type(policy[key]) not in (int, float) or not math.isfinite(policy[key]) or policy[key] < 0):
            raise ValueError("invalid host-state limit " + key)
    for key in ("ci_uids", "ignore_pids", "normal_pressure_levels"):
        if key in policy and (not isinstance(policy[key], (list, tuple)) or any(type(value) is not int or value < 0 for value in policy[key])):
            raise ValueError("invalid host-state identity/pressure policy")
    if "require_pressure_level" in policy and type(policy["require_pressure_level"]) is not bool:
        raise ValueError("require_pressure_level must be boolean")


def _observer_ancestors(value):
    """Exclude only this sample's observer and its recorded parent chain."""
    parents = {row["pid"]: row["ppid"] for row in value["processes"]}
    pid, found = value.get("observer_pid"), set()
    while type(pid) is int and pid > 0 and pid not in found:
        found.add(pid)
        pid = parents.get(pid)
    return found


def compare_snapshots(before, after, policy, *, timed=False):
    """Assess idle thresholds, or flag timed swap and compression with pressure.

    CPU cores are cumulative CPU milliseconds divided by observed wall time.
    Exited processes have unknown final CPU; their disappearance alone is not a
    failure. New processes' observed lifetime CPU is included conservatively.
    """
    _policy(policy)
    result = dict(status="failed", mode="timed" if timed else "quiet", reasons=[], flags={})
    reasons = result["reasons"]
    if before.get("status") != "passed" or after.get("status") != "passed":
        reasons.append("snapshot-unavailable")
        return result
    elapsed = (after["finished_monotonic_ns"] - before["finished_monotonic_ns"]) / 1e9
    if (elapsed <= 0 or before["vm_stat"]["page_size_bytes"] != after["vm_stat"]["page_size_bytes"]
            or before.get("memory_bytes") != after.get("memory_bytes")):
        reasons.append("snapshot-time-or-memory-identity-changed")
        return result
    page_size = after["vm_stat"]["page_size_bytes"]
    delta = {key: after["vm_stat"]["counters"][key] - before["vm_stat"]["counters"][key] for key in COUNTERS}
    if any(value < 0 for value in delta.values()):
        reasons.append("vm-counter-reset")
        return result
    delta_bytes = {key: value * page_size for key, value in delta.items()}
    # A swap-out means memory ran short in the window; a swap-in only brings back a page
    # put out at another time, and every count stays in counter_delta_bytes.
    swapout = delta_bytes["swapouts"]
    # Likewise a compression means memory ran short; a decompression only brings
    # back a page compressed at another time.
    compressed = delta_bytes["compressions"]
    levels = [value.get("memory_pressure_level") for value in (before, after)]
    pressure_unknown = None in levels and policy.get("require_pressure_level", False)
    pressure_bad = any(level is not None and level not in policy.get("normal_pressure_levels", [level]) for level in levels)
    compression_large = compressed > policy.get("max_compressed_bytes", 0)
    flags = dict(swapout_activity=swapout > 0, compression_activity=compressed > 0,
        pagein_activity=delta["pageins"] > 0, pageout_activity=delta["pageouts"] > 0,
        swap_usage_changed=before["swap"]["used_bytes"] != after["swap"]["used_bytes"],
        memory_pressure_changed=levels[0] != levels[1], memory_pressure_abnormal=pressure_bad,
        compression_with_pressure=compression_large and pressure_bad)
    old = {row["pid"]: row for row in before["processes"]}
    current = {row["pid"]: row for row in after["processes"]}
    ignored = set(policy.get("ignore_pids", [])) | _observer_ancestors(before) | _observer_ancestors(after)
    ci_uids = set(policy.get("ci_uids", []))
    activity, changed = [], []
    for pid, row in current.items():
        if pid in ignored or row["uid"] in ci_uids:
            continue
        prior = old.get(pid)
        same = prior is not None and all(prior[key] == row[key] for key in ("uid", "comm")) and row["elapsed_ms"] >= prior["elapsed_ms"]
        if prior is not None and (not same or row["cpu_time_ms"] < prior["cpu_time_ms"]):
            changed.append(pid)
            same = False
        cpu = row["cpu_time_ms"] - prior["cpu_time_ms"] if same else row["cpu_time_ms"]
        activity.append(dict(pid=pid, uid=row["uid"], name=row["name"], comm=row["comm"],
            cpu_delta_ms=cpu, cpu_cores=cpu / (elapsed * 1000), rss_bytes=row["rss_bytes"]))
    activity.sort(key=lambda row: (-row["cpu_cores"], row["pid"]))
    # Preserve the native Python 3.9 accumulation order when replaying evidence.
    # Built-in float sum changed in 3.12, breaking exact retained comparisons.
    total_cpu = 0
    for row in activity:
        total_cpu += row["cpu_cores"]
    max_cpu = max((row["cpu_cores"] for row in activity), default=0)
    max_rss = max((row["rss_bytes"] for row in activity), default=0)
    # Available as the kernel counts it: free pages plus file-backed pages, which
    # the system reclaims without compression or swap. A model mapped by the
    # previous run stays cached on the active queue until pressure ages it, so
    # free plus inactive would refuse every run after the first.
    available = [sum(value["vm_stat"]["pages"][key] for key in ("free", "file_backed")) * page_size for value in (before, after)]
    result.update(elapsed_s=elapsed, counter_delta_pages=delta, counter_delta_bytes=delta_bytes,
        swapout_bytes=swapout, compressed_bytes=compressed,
        swap_used_delta_bytes=after["swap"]["used_bytes"] - before["swap"]["used_bytes"],
        memory_pressure_levels=levels, available_bytes=available, flags=flags,
        process_activity=dict(non_ci_cpu_cores=total_cpu, max_non_ci_process_cpu_cores=max_cpu,
            max_non_ci_process_rss_bytes=max_rss, processes=activity,
            ignored_pids=sorted(ignored),
            started_pids=sorted(current.keys() - old.keys()), exited_pids=sorted(old.keys() - current.keys()),
            identity_changed_pids=changed))
    if pressure_unknown:
        reasons.append("memory-pressure-unavailable")
    if timed:
        if flags["swapout_activity"]:
            reasons.append("swapout-activity")
        if flags["compression_with_pressure"]:
            reasons.append("compression-with-abnormal-pressure")
        if pressure_bad:
            reasons.append("memory-pressure-abnormal")
        for key, value in (("max_non_ci_cpu_cores", total_cpu), ("max_non_ci_process_cpu_cores", max_cpu)):
            if key in policy and value > policy[key]:
                reasons.append(key + "-exceeded")
        result["status"] = "failed" if pressure_unknown else "flagged" if reasons else "passed"
    else:
        if pressure_bad:
            reasons.append("memory-pressure-abnormal")
        checks = {"max_swapout_bytes": swapout, "max_compressed_bytes": compressed,
            "max_non_ci_cpu_cores": total_cpu, "max_non_ci_process_cpu_cores": max_cpu,
            "max_non_ci_process_rss_bytes": max_rss}
        reasons.extend(key + "-exceeded" for key, value in checks.items() if key in policy and value > policy[key])
        if "min_available_bytes" in policy and min(available) < policy["min_available_bytes"]:
            reasons.append("available-below-minimum")
        result["status"] = "failed" if reasons else "passed"
    return result


def quiet_check(policy, *, sample_interval_s=2, timeout_s=5, max_output_bytes=2 << 20):
    """Exactly two observations and one bounded interval; no retries or cleanup."""
    _policy(policy)
    if type(sample_interval_s) not in (int, float) or not 0 < sample_interval_s <= 10:
        raise ValueError("bounded positive sample interval required")
    before = snapshot(timeout_s=timeout_s, max_output_bytes=max_output_bytes)
    time.sleep(sample_interval_s)
    after = snapshot(timeout_s=timeout_s, max_output_bytes=max_output_bytes)
    return dict(before=before, after=after, assessment=compare_snapshots(before, after, policy))
