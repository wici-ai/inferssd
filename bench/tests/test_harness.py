import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

BENCH = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


gate = module("device_gate", BENCH / "device/gate.py")
driver = module("abba_driver", BENCH / "run_abba.py")


@pytest.fixture
def setup(tmp_path):
    arm = tmp_path / "arm.py"
    arm.write_text('''import json, os, pathlib, signal, sys, time
out = pathlib.Path(os.environ['INFERSSD_RUN_DIR'])
(out / 'pid').write_text(str(os.getpid()))
mode = sys.argv[1]
if mode == 'wait':
    def stop(*_):
        (out / 'terminated').write_text('owned arm stopped')
        raise SystemExit(7)
    signal.signal(signal.SIGTERM, stop)
    (out / 'ready').touch()
    time.sleep(30)
if mode == 'fail':
    raise SystemExit(9)
if mode != 'missing':
    result = {'input_tokens': 614, 'output_tokens': 96, 'prompt_sha256': 'a' * 64,
              'metrics': {'ttft_s': 1 if os.environ['INFERSSD_ARM'] == 'A' else 2}}
    if mode == 'wrong': result['input_tokens'] = 613
    pathlib.Path(os.environ['INFERSSD_RESULT_PATH']).write_text(json.dumps(result))
''')
    config = {"protocol": {"input_tokens": 614, "output_tokens": 96, "prompt_sha256": "a" * 64},
              "arms": {label: {"revision": label * 40, "argv": [sys.executable, str(arm), "ok"]}
                       for label in ("A", "B")}}
    return tmp_path, config


def command(root, config, *extra):
    path = root / "config.json"
    path.write_text(json.dumps(config))
    return [sys.executable, str(BENCH / "run_abba.py"), "--config", str(path),
            "--output", str(root / "run"), "--lock", str(root / "device.lock"),
            "--stop", str(root / "STOP"), *extra]


def read_rows(root):
    return [json.loads(line) for line in (root / "run/arms.jsonl").read_text().splitlines()]


def test_complete_abba_keeps_all_four_values(setup):
    root, config = setup
    subprocess.run(command(root, config), check=True)
    rows = read_rows(root)
    assert [row["arm"] for row in rows] == ["A", "B", "B", "A"]
    summary = json.loads((root / "run/summary.json").read_text())
    assert summary["complete"] and summary["recorded_arms"] == 4
    assert summary["metrics"]["ttft_s"] == {
        "raw": {"A": [1, 1], "B": [2, 2]}, "mean": {"A": 1, "B": 2}, "B_over_A": 2}


def test_existing_flock_is_preserved_and_never_released(setup):
    root, config = setup
    path = root / "device.lock"
    path.write_text("existing owner metadata\n")
    inode = path.stat().st_ino
    with path.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(command(root, config), capture_output=True)
        assert result.returncode == 2
        assert b"lock is held" in result.stderr
        assert not (root / "run").exists()
        assert path.stat().st_ino == inode and path.read_text() == "existing owner metadata\n"
    subprocess.run(command(root, config), check=True)
    assert path.stat().st_ino == inode
    assert path.read_text() == "existing owner metadata\n"


def test_preexisting_stop_starts_nothing(setup):
    root, config = setup
    (root / "STOP").touch()
    result = subprocess.run(command(root, config), capture_output=True)
    assert result.returncode == 2 and not (root / "run").exists()


def test_active_stop_terminates_owned_arm_and_records_rest(setup):
    root, config = setup
    config["arms"]["A"]["argv"][-1] = "wait"
    process = subprocess.Popen(command(root, config))
    try:
        ready = root / "run/01-A/ready"
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        (root / "STOP").touch()
        assert process.wait(timeout=8) == 1
        assert (root / "run/01-A/terminated").exists()
        assert [row["status"] for row in read_rows(root)] == ["failed", "not_run", "not_run", "not_run"]
        with (root / "device.lock").open() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=8)


@pytest.mark.parametrize("mode", ["fail", "missing", "wrong"])
def test_bad_second_arm_stops_without_retry_or_filtering(setup, mode):
    root, config = setup
    config["arms"]["B"]["argv"][-1] = mode
    result = subprocess.run(command(root, config), capture_output=True)
    assert result.returncode == 1
    rows = read_rows(root)
    assert [row["status"] for row in rows] == ["passed", "failed", "not_run", "not_run"]
    assert len(rows) == 4 and not (root / "run/03-B").exists()
    assert not json.loads((root / "run/summary.json").read_text())["complete"]


def test_timeout_is_failure_not_a_filtered_slow_value(setup):
    root, config = setup
    config["arms"]["A"]["argv"][-1] = "wait"
    result = subprocess.run(command(root, config, "--arm-timeout", "0.3"), capture_output=True)
    assert result.returncode == 1
    assert read_rows(root)[0]["error"] == "arm timeout"


def test_termination_waits_for_children_after_group_leader_exits(tmp_path):
    child = tmp_path / "child.py"
    child.write_text("import os, pathlib, signal, sys, time\n"
                     "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                     "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
                     "time.sleep(30)\n")
    pid_file = tmp_path / "child.pid"
    parent = subprocess.Popen([sys.executable, "-c",
                               "import subprocess, sys, time; "
                               "subprocess.Popen(sys.argv[1:]); time.sleep(30)",
                               sys.executable, str(child), str(pid_file)], start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert pid_file.exists()
        driver.terminate_owned_group(parent)
        assert parent.poll() is not None
        state = subprocess.run(["ps", "-o", "stat=", "-p", pid_file.read_text()],
                               capture_output=True, text=True).stdout.strip()
        assert not state or state.startswith("Z")
    finally:
        try:
            os.killpg(parent.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        parent.wait(timeout=5)


def test_successful_wrapper_with_orphan_fails_before_unlocking_or_next_arm(setup):
    root, config = setup
    wrapper = root / "orphan-wrapper.py"
    wrapper.write_text('''import json, os, pathlib, subprocess, sys, time
out = pathlib.Path(os.environ['INFERSSD_RUN_DIR'])
subprocess.Popen([sys.executable, '-c',
    'import os, pathlib, signal, sys, time; '
    'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
    'pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)',
    str(out / 'child.pid')])
while not (out / 'child.pid').exists(): time.sleep(.01)
pathlib.Path(os.environ['INFERSSD_RESULT_PATH']).write_text(json.dumps({
    'input_tokens': 614, 'output_tokens': 96, 'prompt_sha256': 'a' * 64,
    'metrics': {'ttft_s': 1}}))
''')
    config["arms"]["A"]["argv"] = [sys.executable, str(wrapper)]
    process = subprocess.Popen(command(root, config))
    child_pid = None
    try:
        pid_file = root / "run/01-A/child.pid"
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert pid_file.exists()
        child_pid = int(pid_file.read_text())
        with (root / "device.lock").open() as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert process.wait(timeout=8) == 1
        rows = read_rows(root)
        assert [row["status"] for row in rows] == ["failed", "not_run", "not_run", "not_run"]
        assert rows[0]["returncode"] == 0 and "descendants remain" in rows[0]["error"]
        assert not (root / "run/02-B").exists()
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(child_pid)],
                               capture_output=True, text=True).stdout.strip()
        assert not state or state.startswith("Z")
        with (root / "device.lock").open() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=8)
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_long_request_is_not_started_by_default(setup):
    root, config = setup
    config["protocol"]["input_tokens"] = 4096
    result = subprocess.run(command(root, config), capture_output=True)
    assert result.returncode == 2 and not (root / "run").exists()


def test_gate_requires_complete_exact_statuses():
    baseline = {"tests": {"ok": {"expect": "passed"},
                          "known": {"expect": "failed", "reason": "tracked numerical defect"}}}
    assert gate.compare({"ok": "passed", "known": "failed"}, baseline)["passed"]
    for actual in ({"ok": "failed", "known": "failed"}, {"ok": "passed"},
                   {"ok": "passed", "known": "passed"}, {"ok": "skipped", "known": "failed"},
                   {"ok": "passed", "known": "failed", "new": "failed"}):
        assert not gate.compare(actual, baseline)["passed"]


def test_gate_skips_need_reason_and_execution():
    for reason in (None, "", " ", 1):
        with pytest.raises(ValueError):
            gate.compare({"x": "skipped"}, {"tests": {"x": {"expect": "skipped", "reason": reason}}})
    assert not gate.compare({"x": "skipped"}, {"tests": {
        "x": {"expect": "skipped", "reason": "explicitly disabled"}}})["passed"]


def test_junit_reader_keeps_failure_skip_and_pass(tmp_path):
    path = tmp_path / "results.xml"
    path.write_text('<testsuite><testcase name="one"/><testcase name="two"><failure/></testcase>'
                    '<testcase name="three"><skipped/></testcase></testsuite>')
    assert gate.read_results(path) == {"one": "passed", "two": "failed", "three": "skipped"}


@pytest.mark.parametrize("rows", [[], [{"name": "x", "status": "unknown"}],
    [{"name": "x", "status": "passed"}, {"name": "x", "status": "passed"}]])
def test_reader_rejects_zero_unknown_and_duplicate_results(tmp_path, rows):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"tests": rows}))
    with pytest.raises(ValueError):
        gate.read_results(path)


def test_real_ctest_return77_is_skipped_not_passed(tmp_path):
    # Exercise CTest's real JUnit contract, not a hand-written <skipped> fixture.
    import shutil
    cmake, ctest = shutil.which("cmake"), shutil.which("ctest")
    if not cmake or not ctest:
        pytest.skip("CMake/CTest unavailable for the JUnit integration fixture")
    (tmp_path / "CMakeLists.txt").write_text(
        'cmake_minimum_required(VERSION 3.16)\nproject(skip_contract NONE)\nenable_testing()\n'
        f'add_test(NAME resource COMMAND "{sys.executable}" -c "raise SystemExit(77)")\n'
        'set_tests_properties(resource PROPERTIES SKIP_RETURN_CODE 77)\n'
        f'add_test(NAME ordinary_failure COMMAND "{sys.executable}" -c "raise SystemExit(1)")\n')
    build = tmp_path / "build"
    subprocess.run([cmake, "-S", str(tmp_path), "-B", str(build)], check=True, capture_output=True)
    junit = tmp_path / "results.xml"
    result = subprocess.run([ctest, "--test-dir", str(build), "--output-junit", str(junit)],
                            check=False, capture_output=True, timeout=20)
    assert result.returncode != 0  # the ordinary failure is never swallowed
    actual = gate.read_results(junit)
    assert actual == {"resource": "skipped", "ordinary_failure": "failed"}
    report = gate.compare(actual, {"tests": {
        "resource": {"expect": "passed"},
        "ordinary_failure": {"expect": "failed", "reason": "negative control"}}})
    assert not report["passed"] and report["counts"]["skipped"] == 1
    all_skipped = gate.compare({"resource": actual["resource"]}, {"tests": {
        "resource": {"expect": "skipped", "reason": "missing runtime prerequisite"}}})
    assert not all_skipped["passed"] and {"error": "zero executed tests"} in all_skipped["differences"]
