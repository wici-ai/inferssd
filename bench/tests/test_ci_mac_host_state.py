"""Synthetic Darwin observations and bounded helper commands; no model runs."""

import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci import mac_host_state as host


VM_STAT = '''Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               1000.
Pages active:                             9000.
Pages inactive:                           2000.
Pages speculative:                         100.
Pages throttled:                             0.
Pages wired down:                         3000.
Pages purgeable:                           200.
"Translation faults":                 123456789.
Pages copy-on-write:                      10000.
Pages zero filled:                      9000000.
Pages reactivated:                      8000000.
Pages purged:                            100000.
File-backed pages:                        10000.
Anonymous pages:                           9000.
Pages stored in compressor:              100000.
Pages occupied by compressor:             20000.
Decompressions:                        20000000.
Compressions:                          30000000.
Pageins:                                4000000.
Pageouts:                               1000000.
Swapins:                                5000000.
Swapouts:                               6000000.
'''
SWAP = "total = 4096.00M  used = 128.50M  free = 3967.50M  (encrypted)\n"
PROCESSES = """  111     1   501    52:13.45  01-02:03:04    1024 /Applications/Existing Service.app/Contents/MacOS/Existing Service
  222   111   501  1:02:03.004        03:05    2048 /usr/bin/another
"""


def observation(finished_s=1):
    return dict(status="passed", finished_monotonic_ns=int(finished_s * 1e9),
                vm_stat=host.parse_vm_stat(VM_STAT), swap=host.parse_swap(SWAP),
                memory_bytes=64 * 1024 ** 3,
                memory_pressure_level=1, processes=host.parse_processes(PROCESSES))


def pair():
    before, after = observation(1), observation(3)
    for row in after["processes"]:
        row["elapsed_ms"] += 2000
    return before, after


def command_result(stdout, **changes):
    result = dict(stdout=stdout, stderr="", returncode=0, error=None, elapsed_s=0.01)
    result.update(changes)
    return result


def test_parse_real_darwin_vm_stat_and_swap_units():
    vm = host.parse_vm_stat(VM_STAT)
    assert vm["page_size_bytes"] == 16384
    assert vm["pages"]["free"] == 1000 and vm["pages"]["inactive"] == 2000
    assert vm["pages"]["file_backed"] == 10000
    assert vm["pages"]["occupied_by_compressor"] == 20000
    assert vm["counters"] == dict(swapins=5000000, swapouts=6000000,
                                  compressions=30000000, decompressions=20000000,
                                  pageins=4000000, pageouts=1000000)
    assert host.parse_swap(SWAP) == dict(total_bytes=4096 * 1024 ** 2,
                                       used_bytes=128 * 1024 ** 2 + 512 * 1024,
                                       free_bytes=3967 * 1024 ** 2 + 512 * 1024)


def test_parse_ps_cpu_elapsed_and_command_paths_with_spaces():
    first, second = host.parse_processes(PROCESSES)
    assert first == dict(pid=111, ppid=1, uid=501, cpu_time_ms=3133450,
                        elapsed_ms=93784000, rss_bytes=1024 ** 2,
                        comm="/Applications/Existing Service.app/Contents/MacOS/Existing Service",
                        name="Existing Service")
    assert second["cpu_time_ms"] == 3723004 and second["elapsed_ms"] == 185000
    assert second["rss_bytes"] == 2 * 1024 ** 2


@pytest.mark.parametrize("parser,raw", [
    (host.parse_vm_stat, VM_STAT.replace("Swapouts:", "Missing:")),
    (host.parse_vm_stat, VM_STAT.replace("16384 bytes", "0 bytes")),
    (host.parse_swap, "total = 1.00M used = 2.00M free = 0.00M"),
    (host.parse_swap, "total = 1.00M used = 0.00M"),
    (host.parse_processes, ""),
    (host.parse_processes, PROCESSES + PROCESSES.splitlines()[0]),
    (host.parse_processes, "111 1 501 0:60.00 00:01 20 /usr/bin/a"),
    (host.parse_processes, "111 1 501 0:00.00 00:01 -1 /usr/bin/a"),
    (host.parse_processes, "111 1 501 0:00.00 00:01 20"),
])
def test_incomplete_or_invalid_command_evidence_is_rejected(parser, raw):
    with pytest.raises(ValueError):
        parser(raw)


def test_existing_services_and_historical_counters_do_not_fail_quiet_check():
    before, after = pair()
    after["swap"]["used_bytes"] -= 1024 ** 2
    result = host.compare_snapshots(before, after, dict(max_swap_delta_bytes=0,
        max_compression_delta_bytes=0, max_non_ci_cpu_cores=0))
    assert result["status"] == "passed" and result["reasons"] == []
    assert result["swap_delta_bytes"] == result["compression_delta_bytes"] == 0
    assert result["swap_used_delta_bytes"] == -1024 ** 2
    assert len(result["process_activity"]["processes"]) == 2
    assert result["process_activity"]["non_ci_cpu_cores"] == 0


def test_cumulative_page_deltas_use_page_size_not_current_swap_usage():
    before, after = pair()
    increments = dict(swapins=2, swapouts=3, compressions=4, decompressions=5,
                      pageins=6, pageouts=7)
    for key, value in increments.items():
        after["vm_stat"]["counters"][key] += value
    result = host.compare_snapshots(before, after, {})
    assert result["counter_delta_pages"] == increments
    assert result["counter_delta_bytes"] == {key: value * 16384 for key, value in increments.items()}
    assert result["swap_delta_bytes"] == 5 * 16384
    assert result["compression_delta_bytes"] == 9 * 16384
    assert result["swap_used_delta_bytes"] == 0


@pytest.mark.parametrize("fault,reason", [
    ("counter", "vm-counter-reset"),
    ("clock", "snapshot-time-or-memory-identity-changed"),
    ("page-size", "snapshot-time-or-memory-identity-changed"),
    ("memory-size", "snapshot-time-or-memory-identity-changed"),
])
def test_counter_reset_or_incomparable_snapshots_fail_closed(fault, reason):
    before, after = pair()
    if fault == "counter":
        after["vm_stat"]["counters"]["pageins"] -= 1
    elif fault == "clock":
        after["finished_monotonic_ns"] = before["finished_monotonic_ns"]
    elif fault == "page-size":
        after["vm_stat"]["page_size_bytes"] = 4096
    else:
        after["memory_bytes"] //= 2
    for timed in (False, True):
        result = host.compare_snapshots(before, after, {}, timed=timed)
        assert result["status"] == "failed" and result["reasons"] == [reason]


@pytest.mark.parametrize("limit,value,reason", [
    ("max_non_ci_cpu_cores", 0.6, "max_non_ci_cpu_cores-exceeded"),
    ("max_non_ci_process_cpu_cores", 0.3, "max_non_ci_process_cpu_cores-exceeded"),
    ("max_non_ci_process_rss_bytes", 1024 ** 2, "max_non_ci_process_rss_bytes-exceeded"),
])
def test_total_cpu_single_cpu_and_rss_limits_are_independent(limit, value, reason):
    before, after = pair()
    after["processes"][0]["cpu_time_ms"] += 600
    after["processes"][1]["cpu_time_ms"] += 700
    result = host.compare_snapshots(before, after, {limit: value})
    assert result["status"] == "failed" and result["reasons"] == [reason]
    activity = result["process_activity"]
    assert activity["non_ci_cpu_cores"] == pytest.approx(0.65)
    assert activity["max_non_ci_process_cpu_cores"] == pytest.approx(0.35)
    assert activity["max_non_ci_process_rss_bytes"] == 2 * 1024 ** 2
    assert [row["pid"] for row in activity["processes"]] == [222, 111]


def test_quiet_threshold_equality_passes_and_low_memory_either_sample_fails():
    before, after = pair()
    after["processes"][0]["cpu_time_ms"] += 1000
    after["vm_stat"]["counters"]["swapins"] += 1
    after["vm_stat"]["counters"]["compressions"] += 1
    policy = dict(max_non_ci_cpu_cores=0.5, max_non_ci_process_cpu_cores=0.5,
                  max_non_ci_process_rss_bytes=2 * 1024 ** 2,
                  max_swap_delta_bytes=16384, max_compression_delta_bytes=16384,
                  min_free_inactive_bytes=3000 * 16384)
    assert host.compare_snapshots(before, after, policy)["status"] == "passed"
    before["vm_stat"]["pages"]["free"] -= 1
    result = host.compare_snapshots(before, after, policy)
    assert result["reasons"] == ["free-inactive-below-minimum"]


@pytest.mark.parametrize("changed", ["uid", "comm", "elapsed_ms", "cpu_time_ms"])
def test_restarted_or_reused_pid_counts_current_lifetime_cpu(changed):
    before, after = pair()
    row = after["processes"][0]
    if changed == "uid":
        row["uid"] = 502
    elif changed == "comm":
        row["comm"] = "/usr/bin/replacement"
    elif changed == "elapsed_ms":
        row["elapsed_ms"] = 100
    else:
        row["cpu_time_ms"] = 100
    result = host.compare_snapshots(before, after, {})
    activity = result["process_activity"]
    assert activity["identity_changed_pids"] == [111]
    observed = next(item for item in activity["processes"] if item["pid"] == 111)
    assert observed["cpu_delta_ms"] == row["cpu_time_ms"]
    assert observed["cpu_cores"] == row["cpu_time_ms"] / 2000


def test_new_process_lifetime_cpu_is_counted_and_exit_alone_is_not_failure():
    before, after = pair()
    after["processes"] = [dict(after["processes"][1], pid=333, cpu_time_ms=100)]
    result = host.compare_snapshots(before, after, dict(max_non_ci_cpu_cores=0.1))
    assert result["status"] == "passed"
    activity = result["process_activity"]
    assert activity["started_pids"] == [333] and activity["exited_pids"] == [111, 222]
    assert activity["non_ci_cpu_cores"] == 0.05


def test_ci_uids_and_ignored_pids_are_excluded_from_cpu_and_rss_limits():
    before, after = pair()
    for snapshots in (before, after):
        snapshots["processes"][0]["uid"] = 62002
    for row in after["processes"]:
        row["cpu_time_ms"] += 10000
        row["rss_bytes"] = 80 * 1024 ** 3
    result = host.compare_snapshots(before, after, dict(ci_uids=[62002], ignore_pids=[222],
        max_non_ci_cpu_cores=0, max_non_ci_process_cpu_cores=0, max_non_ci_process_rss_bytes=0))
    assert result["status"] == "passed"
    assert result["process_activity"]["processes"] == []
    assert result["process_activity"]["max_non_ci_process_rss_bytes"] == 0


def test_both_observer_parent_chains_are_excluded_but_other_services_remain():
    before, after = pair()
    for sample in (before, after):
        for pid, ppid in ((90, 1), (91, 1), (100, 90), (101, 91)):
            sample["processes"].append(dict(sample["processes"][0], pid=pid, ppid=ppid))
    before["observer_pid"], after["observer_pid"] = 100, 101
    for row in after["processes"]:
        row["cpu_time_ms"] += 10000 if row["pid"] != 111 else 100
    result = host.compare_snapshots(before, after, dict(ignore_pids=[222], max_non_ci_cpu_cores=0.1))
    assert result["status"] == "passed"
    activity = result["process_activity"]
    assert activity["ignored_pids"] == [1, 90, 91, 100, 101, 222]
    assert [row["pid"] for row in activity["processes"]] == [111]
    assert activity["non_ci_cpu_cores"] == 0.05


@pytest.mark.parametrize("swap,compression,pressure,expected,reasons", [
    (0, 0, 1, "passed", []),
    (0, 2, 1, "passed", []),
    (0, 0, 2, "flagged", ["memory-pressure-abnormal"]),
    (1, 0, 1, "flagged", ["swap-activity"]),
    (0, 2, 2, "flagged", ["compression-with-abnormal-pressure", "memory-pressure-abnormal"]),
    (1, 2, 2, "flagged", ["swap-activity", "compression-with-abnormal-pressure", "memory-pressure-abnormal"]),
])
def test_timed_mode_flags_swap_and_pressure_without_idle_memory_limits(
        swap, compression, pressure, expected, reasons):
    before, after = pair()
    after["vm_stat"]["counters"]["swapouts"] += swap
    after["vm_stat"]["counters"]["compressions"] += compression
    after["memory_pressure_level"] = pressure
    policy = dict(normal_pressure_levels=[1], max_compression_delta_bytes=16384,
                  max_swap_delta_bytes=0, max_non_ci_cpu_cores=0,
                  max_non_ci_process_cpu_cores=0, max_non_ci_process_rss_bytes=0,
                  min_free_inactive_bytes=100 * 1024 ** 3)
    result = host.compare_snapshots(before, after, policy, timed=True)
    assert result["status"] == expected and result["reasons"] == reasons
    assert result["flags"]["swap_activity"] == bool(swap)
    assert result["flags"]["compression_activity"] == bool(compression)
    assert result["flags"]["memory_pressure_abnormal"] == (pressure != 1)
    assert result["flags"]["memory_pressure_changed"] == (pressure != 1)
    assert result["mode"] == "timed"


@pytest.mark.parametrize("limit", ["max_non_ci_cpu_cores", "max_non_ci_process_cpu_cores"])
def test_timed_mode_flags_non_ci_cpu_activity(limit):
    before, after = pair()
    after["processes"][0]["cpu_time_ms"] += 2000
    result = host.compare_snapshots(before, after, {limit: 0.5}, timed=True)
    assert result["status"] == "flagged" and result["reasons"] == [limit + "-exceeded"]


@pytest.mark.parametrize("timed", [False, True])
def test_required_pressure_unavailable_fails_in_both_modes(timed):
    before, after = pair()
    after["memory_pressure_level"] = None
    result = host.compare_snapshots(before, after, dict(require_pressure_level=True), timed=timed)
    assert result["status"] == "failed" and result["reasons"] == ["memory-pressure-unavailable"]


@pytest.mark.parametrize("failure", ["command", "parse", "pressure", "pressure-invalid"])
def test_snapshot_retains_failure_and_collects_all_remaining_raw_evidence(monkeypatch, failure):
    monkeypatch.setattr(host.sys, "platform", "darwin")
    responses = dict(vm_stat=command_result(VM_STAT), swap=command_result(SWAP),
                     memory_bytes=command_result(str(64 * 1024 ** 3)),
                     memory_pressure_level=command_result("1\n"),
                     processes=command_result(PROCESSES))
    source = "memory_pressure_level" if failure.startswith("pressure") else "vm_stat"
    if failure == "parse":
        responses[source] = command_result("malformed vm output")
    elif failure == "pressure-invalid":
        responses[source] = command_result("-1\n")
    else:
        responses[source] = command_result("partial", returncode=1, stderr="unavailable", error="read failed")
    calls = []
    def command(argv, timeout_s, max_output_bytes):
        calls.append((argv, timeout_s, max_output_bytes))
        key = next(key for key, value in host.COMMANDS.items() if value == argv)
        return copy.deepcopy(responses[key])
    monkeypatch.setattr(host, "_bounded_command", command)
    result = host.snapshot(timeout_s=0.5, max_output_bytes=2048)
    assert calls == [(argv, 0.5, 2048) for argv in host.COMMANDS.values()]
    assert result["raw"] == responses
    assert result["processes"] == host.parse_processes(PROCESSES)
    assert result["memory_bytes"] == 64 * 1024 ** 3
    assert result["finished_monotonic_ns"] >= result["started_monotonic_ns"]
    if failure.startswith("pressure"):
        assert result["status"] == "passed" and result["errors"] == []
        assert result["unavailable"][0]["source"] == source
        assert result["memory_pressure_level"] is None
    else:
        assert result["status"] == "failed" and result["errors"][0]["source"] == source


@pytest.mark.parametrize("failed", [False, True])
def test_quiet_check_takes_exactly_two_snapshots_and_one_interval_without_retry(monkeypatch, failed):
    before, after = pair()
    if failed:
        before["status"] = "failed"
    samples = iter((before, after))
    events = []
    def snapshot(**kwargs):
        events.append(("snapshot", kwargs))
        return next(samples)
    monkeypatch.setattr(host, "snapshot", snapshot)
    monkeypatch.setattr(host.time, "sleep", lambda interval: events.append(("sleep", interval)))
    result = host.quiet_check({}, sample_interval_s=0.25, timeout_s=0.5, max_output_bytes=2048)
    assert events == [("snapshot", dict(timeout_s=0.5, max_output_bytes=2048)),
                      ("sleep", 0.25),
                      ("snapshot", dict(timeout_s=0.5, max_output_bytes=2048))]
    assert result["before"] is before and result["after"] is after
    assert result["assessment"]["status"] == ("failed" if failed else "passed")
    if failed:
        assert result["assessment"]["reasons"] == ["snapshot-unavailable"]


@pytest.mark.parametrize("kwargs", [
    dict(timeout_s=0), dict(timeout_s=11), dict(timeout_s=float("inf")),
    dict(timeout_s=float("nan")), dict(timeout_s=True),
    dict(max_output_bytes=1023), dict(max_output_bytes=(4 << 20) + 1),
    dict(max_output_bytes=True), dict(max_output_bytes=2048.0),
])
def test_snapshot_rejects_unbounded_or_invalid_command_limits_before_launch(monkeypatch, kwargs):
    monkeypatch.setattr(host, "_bounded_command", lambda *args: pytest.fail("must validate before launch"))
    with pytest.raises(ValueError, match="bounds"):
        host.snapshot(**kwargs)


@pytest.mark.parametrize("interval", [0, -1, 11, True, float("inf"), float("nan")])
def test_quiet_check_rejects_unbounded_intervals_before_observation(monkeypatch, interval):
    monkeypatch.setattr(host, "snapshot", lambda **kwargs: pytest.fail("must validate before observation"))
    with pytest.raises(ValueError, match="interval"):
        host.quiet_check({}, sample_interval_s=interval)


def test_bounded_command_retains_partial_output_on_timeout():
    result = host._bounded_command(
        [sys.executable, "-c", "import time; print('partial', flush=True); time.sleep(5)"],
        timeout_s=0.25, max_output_bytes=1024)
    assert result["error"] and "deadline" in result["error"]
    assert result["stdout"] == "partial\n" and result["returncode"] != 0
    assert result["elapsed_s"] < 3


def test_bounded_command_caps_combined_stdout_and_stderr():
    result = host._bounded_command(
        [sys.executable, "-c", "import os; os.write(1, b'a' * 40); os.write(2, b'b' * 40)"],
        timeout_s=2, max_output_bytes=64)
    assert result["error"] and "output byte limit" in result["error"]
    assert len(result["stdout"].encode()) + len(result["stderr"].encode()) == 64
    assert result["returncode"] is not None


def test_fixed_process_command_requests_executable_name_without_argv_or_environment():
    assert host.COMMANDS["processes"] == (
        "/bin/ps", "-axo", "pid=,ppid=,uid=,time=,etime=,rss=,comm=")
