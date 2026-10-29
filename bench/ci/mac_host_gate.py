"""Fixed Mac measurement conditions, retained separately from engine settings.

This trusted coordinator only observes the host. It never closes applications,
changes swap, clears caches, retries a completion, or attributes pressure to a PR.
"""

import copy
import json
from pathlib import Path

from ci import mac_host_state as state
from ci.contract import digest, load, save


QUIET = "host-quiet.json"
SNAPSHOTS = ("host-state-before-request.json", "host-state-after-request.json",
             "host-state-after-cleanup.json")


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def policy(config):
    """Derive one versioned policy from the sealed physical RAM and budget."""
    host = config["host"]
    total, budget = host["hardware"]["memory_total_bytes"], host["resources"]["memory_budget_bytes"]
    if any(type(value) is not int or value <= 0 for value in (total, budget)) or budget > total:
        raise ValueError("Mac host gate requires sealed physical RAM and memory budget")
    return dict(schema_version=1, memory_total_bytes=total, sample_interval_s=2,
        pair_free_tolerance_bytes=1 << 30, limits=dict(
            ci_uids=[62001, 62002], ignore_pids=[], normal_pressure_levels=[1], require_pressure_level=True,
            min_free_inactive_bytes=min(budget, total * 3 // 4), max_swap_delta_bytes=0,
            max_compression_delta_bytes=64 << 20, max_non_ci_cpu_cores=0.5,
            max_non_ci_process_cpu_cores=0.25, max_non_ci_process_rss_bytes=512 << 20))


def _snapshot(value, expected_memory=None):
    """Reparse retained raw commands; do not trust normalized fields alone."""
    if (not isinstance(value, dict) or value.get("schema_version") != 1
            or value.get("platform") != "darwin" or value.get("status") != "passed"
            or value.get("errors") != [] or value.get("unavailable") != []
            or any(type(value.get(key)) is not int or value[key] <= 0 for key in (
                "observer_pid", "started_monotonic_ns", "finished_monotonic_ns"))
            or value["finished_monotonic_ns"] < value["started_monotonic_ns"]):
        raise ValueError("Mac host snapshot is missing or unavailable")
    parsers = dict(vm_stat=state.parse_vm_stat, swap=state.parse_swap, processes=state.parse_processes,
        memory_bytes=lambda raw: int(raw.strip()), memory_pressure_level=lambda raw: int(raw.strip()))
    raw = value.get("raw")
    if not isinstance(raw, dict) or set(raw) != set(parsers):
        raise ValueError("Mac host raw command inventory differs")
    for name, parse in parsers.items():
        record = raw[name]
        if (not isinstance(record, dict) or type(record.get("returncode")) is not int
                or record["returncode"] != 0 or record.get("error") is not None
                or not isinstance(record.get("stdout"), str) or not isinstance(record.get("stderr"), str)
                or not _same(parse(record["stdout"]), value.get(name))):
            raise ValueError("Mac host raw/parsed observation differs: " + name)
    if (type(value["memory_bytes"]) is not int or value["memory_bytes"] <= 0
            or expected_memory is not None and value["memory_bytes"] != expected_memory):
        raise ValueError("Mac host physical RAM differs from the sealed host")
    return value


def capture(output, name):
    """Keep even a failed collection, then explicitly refuse incomplete evidence."""
    if name not in SNAPSHOTS:
        raise ValueError("fixed Mac host snapshot filename required")
    value = state.snapshot()
    save(Path(output) / name, value)
    return _snapshot(value)


def _quiet(value, expected):
    if (not isinstance(value, dict) or value.get("schema_version") != 1
            or value.get("purpose") != "mac-host-quiet" or not _same(value.get("policy"), expected)):
        raise ValueError("Mac quiet evidence/policy differs")
    before = _snapshot(value["before"], expected["memory_total_bytes"])
    after = _snapshot(value["after"], expected["memory_total_bytes"])
    if after["started_monotonic_ns"] - before["finished_monotonic_ns"] < expected["sample_interval_s"] * 1e9:
        raise ValueError("Mac quiet interval is shorter than the fixed policy")
    assessment = state.compare_snapshots(before, after, expected["limits"])
    if not _same(assessment, value.get("assessment")):
        raise ValueError("Mac quiet assessment differs from retained raw observations")
    if assessment["status"] != "passed":
        raise ValueError("Mac host precondition failed: " + ", ".join(assessment["reasons"]))
    return assessment


def before_launch(output, expected):
    """Two observations, one fixed interval, no retry or model launch on failure."""
    observed = state.quiet_check(expected["limits"], sample_interval_s=expected["sample_interval_s"])
    value = dict(schema_version=1, purpose="mac-host-quiet", policy=expected, **observed)
    save(Path(output) / QUIET, value)
    _quiet(value, expected)
    return value


def _read(folder, name):
    path = Path(folder) / name
    if path.is_symlink() or not path.is_file():
        raise ValueError("Mac host evidence is missing or unsafe: " + name)
    return load(path)


def retained_quiet(folder, expected):
    """Reconstruct the precondition for functional, preflight and timed arms."""
    value = _read(folder, QUIET)
    assessment = _quiet(value, expected)
    return dict(schema_version=1, policy=copy.deepcopy(expected), quiet=assessment,
        prelaunch_free_inactive_bytes=assessment["free_inactive_bytes"][-1],
        artifacts={QUIET: digest(Path(folder) / QUIET)})


def retained_arm(folder, expected):
    """Reconstruct a completed smoke or timed arm from its four raw artifacts."""
    summary = retained_quiet(folder, expected)
    values = [_snapshot(_read(folder, name), expected["memory_total_bytes"]) for name in SNAPSHOTS]
    previous = _read(folder, QUIET)["after"]
    for value in values:
        if value["started_monotonic_ns"] < previous["finished_monotonic_ns"]:
            raise ValueError("Mac host snapshot ordering differs")
        previous = value
    timed = state.compare_snapshots(values[0], values[1], expected["limits"], timed=True)
    cleanup = state.compare_snapshots(values[1], values[2], expected["limits"], timed=True)
    if timed["status"] not in ("passed", "flagged") or cleanup["status"] not in ("passed", "flagged"):
        raise ValueError("Mac timed/cleanup host evidence is unavailable")
    summary.update(timed=timed, cleanup=cleanup)
    summary["artifacts"].update({name: digest(Path(folder) / name) for name in SNAPSHOTS})
    return summary


def cell(summaries):
    """Check every timed run of one cell: {run name: reconstructed arm summary}.

    A flagged run, or prelaunch free memory that differs by more than the free-memory
    tolerance between any two runs, makes an otherwise passing cell needs-review.
    Measured values are never changed, excused or retried.
    """
    if not isinstance(summaries, dict) or len(summaries) < 2:
        raise ValueError("Mac cell check requires every timed run")
    policies = [summary.get("policy") for summary in summaries.values()]
    if any(not _same(policies[0], other) for other in policies[1:]):
        raise ValueError("Mac cell requires matching reconstructed host policies")
    reasons = []
    for name, summary in summaries.items():
        if (summary.get("schema_version") != 1 or summary.get("quiet", {}).get("status") != "passed"
                or summary.get("timed", {}).get("status") not in ("passed", "flagged")
                or summary.get("cleanup", {}).get("status") not in ("passed", "flagged")):
            raise ValueError("Mac cell requires complete reconstructed host evidence")
        if summary["timed"]["status"] == "flagged":
            reasons.extend(name + ":" + reason for reason in summary["timed"]["reasons"])
    free = [summary["prelaunch_free_inactive_bytes"] for summary in summaries.values()]
    spread = max(free) - min(free)
    if spread > policies[0]["pair_free_tolerance_bytes"]:
        reasons.append("prelaunch-free-inactive-difference")
    return dict(schema_version=1, status="needs-review" if reasons else "passed", reasons=reasons,
                runs=copy.deepcopy(summaries), prelaunch_free_spread_bytes=spread,
                interpretation="Sampled host conditions, not isolation or equal-cache proof. Swap/pressure may be model-induced; no automatic attribution to external interference, exemption of measured slowdowns, or retry.")
