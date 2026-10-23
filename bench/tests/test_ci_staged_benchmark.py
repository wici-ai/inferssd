"""Synthetic orchestration only: no native execution or performance claim."""

import base64
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci import staged_benchmark, staged_chat
from ci.contract import digest, load
from ci.stage_session import Stage
from test_final import records


CONFIG = dict(model="muse", revision="a" * 40, argv=["/build/base/bin/inferssd"],
              environment={}, cwd="/build/base", port=18080, timeout_s=600,
              request_timeout_s=300, os_page_cache_treatment="uncontrolled; no purge")


class Session:
    job_id = "a" * 32

    def __init__(self, failure=None):
        self.failure, self.calls, self.started = failure, [], []
        self.completions = 0

    def start(self, name, kind, argv, environment, cwd, timeout):
        assert kind == "server" and timeout == 600
        assert name not in self.started
        self.started.append(name)
        self.completed = False
        return Stage(name, "opaque-handle", 41 + len(self.started), 62002, {"birth": 9}, "owned-scope")

    def call(self, stage, operation, **kwargs):
        self.calls.append((stage.stage_id, operation))
        if operation == "stage.poll":
            return dict(running=self.failure != "startup-exit", returncode=1)
        if operation == "stage.probe":
            value = dict(asdict(stage), ready=True, running=True, executable_sha256="b" * 64)
            if self.failure == "identity" and self.completed:
                value["birth_identity"] = {"birth": 10}
            return value
        if operation == "stage.stop":
            return dict(running=False, returncode=0,
                        receipt={"passed": False, "error": "InterruptedError: stage stopped by trusted controller"},
                        cleanup_confirmed=self.failure != "cleanup")
        assert operation == "stage.client"
        request = kwargs["request"]
        path, chunks = request["path"], []
        if path == "/completion":
            assert request["body"]["n_predict"] == 256
            assert not self.completed
            self.completed = True
            self.completions += 1
            for row in records(count=255 if self.failure == "raw-result" else 256, input_count=600):
                wire = b"data: " + json.dumps(row["event"]).encode() + b"\n\n"
                chunks.append(dict(elapsed_ns=row["monotonic_ns"], data_b64=base64.b64encode(wire).decode()))
        else:
            body = {"status": "ok"}
            if path == "/apply-template":
                body = {"prompt": "rendered", "tokens": [7] * 600}
            elif path == "/props":
                body = {"total_slots": 1, "default_generation_settings": {"n_ctx": 8192}}
            chunks.append(dict(elapsed_ns=1, data_b64=base64.b64encode(json.dumps(body).encode()).decode()))
        return dict(status=200, started_monotonic_ns=100, ended_monotonic_ns=257_000_000_100, chunks=chunks)

    def output(self, stage, stream, offset):
        self.calls.append((stage.stage_id, "output-" + stream))
        assert offset == 0
        if self.failure == "log" and stream == "stdout":
            return b"", 0, False
        return b"native log\n", 11, True


def test_both_arms_share_session_and_revalidate_raw_result(tmp_path):
    session = Session()
    for side in ("base", "candidate"):
        output = tmp_path / side
        result = staged_benchmark.benchmark(session, side, CONFIG, output)
        assert result["status"] == "passed", result
        assert result["completion_requests"] == 1 and result["cleanup_confirmed"]
        assert result["result"] == load(output / "result.json")
        assert result["result"]["metrics"]["decode_first128_tps"] == 1
        assert result["result"]["metrics"]["decode_second128_tps"] == 1
        assert result["result_artifact"]["sha256"] == digest(output / "result.json")
        assert result["runtime_receipt"]["sha256"] == digest(output / "runtime-receipt.json")
        receipt = load(output / "runtime-receipt.json")
        assert receipt["job_id"] == session.job_id
        assert receipt["os_page_cache_treatment"] == CONFIG["os_page_cache_treatment"]
        assert set(receipt["observations"]) == {"started", "health", "health-attempt-0001", "runtime-before", "runtime-after", "stopped"}
        assert result["performance_acceptance"] is False
        assert not (output / "wrapper.json").exists()
        assert load(output / "benchmark.json") == result
    assert session.started == ["base-benchmark-512", "candidate-benchmark-512"]
    assert session.completions == 2


@pytest.mark.parametrize("failure", ["startup-exit", "raw-result", "identity", "cleanup", "log"])
def test_failed_arm_keeps_evidence_and_stops_without_retry(tmp_path, failure):
    session, output = Session(failure), tmp_path / "arm"
    result = staged_benchmark.benchmark(session, "base", CONFIG, output)
    assert result["status"] == "failed"
    assert result["cleanup_confirmed"] is (failure != "cleanup")
    assert len(session.started) == 1
    assert session.completions == (0 if failure == "startup-exit" else 1)
    assert session.calls[-3:] == [("base-benchmark-512", operation) for operation in
                                 ("stage.stop", "output-stdout", "output-stderr")]
    assert (output / "stderr.log").read_bytes() == b"native log\n"
    assert load(output / "stopped.json")["returncode"] == 0
    if failure != "startup-exit":
        assert (output / "result.events.sse").exists()
        assert (output / "runtime-after.json").exists()
    if failure in ("raw-result", "identity", "startup-exit"):
        assert "result" not in result  # raw output never becomes checked evidence
    if failure == "raw-result":
        assert load(output / "result.json")["status"] == "failed"
    assert load(output / "benchmark.json") == result


def test_startup_health_timeout_then_loading_then_ready_is_one_completion(tmp_path, monkeypatch):
    class LoadingSession(Session):
        health_requests = 0

        def call(self, stage, operation, **kwargs):
            response = super().call(stage, operation, **kwargs)
            if operation == "stage.client" and kwargs["request"]["path"] == "/health":
                self.health_requests += 1
                if self.health_requests == 1:
                    return dict(status=0, chunks=[], transport_error="TimeoutError: timed out",
                                started_monotonic_ns=100, ended_monotonic_ns=2_000_000_100)
                if self.health_requests == 2:
                    response["status"] = 503
            return response

    monkeypatch.setattr(staged_benchmark.time, "sleep", lambda _: None)
    session, output = LoadingSession(), tmp_path / "arm"
    result = staged_benchmark.benchmark(session, "base", CONFIG, output)
    assert result["status"] == "passed", result
    assert session.health_requests == 3 and session.completions == 1
    assert [load(path)["status"] for path in sorted(output.glob("health-attempt-*.json"))] == [0, 503, 200]
    assert "timed out" in load(output / "health-attempt-0001.json")["transport_error"]
    assert sum(operation == "stage.poll" for _, operation in session.calls) == 3
    # Readiness tolerance cannot be accidentally used to admit/retry generation.
    with pytest.raises(ValueError, match="only startup health"):
        staged_chat.NativeHTTP(session, None, 18080).request(
            "POST", "/completion", {}, 2, allow_health_timeout=True)


def controlled_stop():
    unit = "inferssd-stage-" + "a" * 32 + ".service"
    cleanup = dict(unit=unit, processes_gone=True,
                   state=dict(MainPID="0", ActiveState="inactive"))
    guardian = dict(unit=unit, returncode=1, controller_disconnected=True,
                    cleanup_confirmed=True, cleanup=cleanup)
    receipt = dict(unit=unit, returncode=1, cleanup_confirmed=True, passed=False,
                   error="InterruptedError: stage stopped by trusted controller",
                   guardian=guardian, cleanup=cleanup)
    return dict(running=False, returncode=1, state="finished", cleanup_confirmed=True,
                scope_identity="/system.slice/" + unit, receipt=receipt)


def test_controlled_launcher_stop_preserves_measurement_failure(tmp_path):
    class ControlledSession(Session):
        def call(self, stage, operation, **kwargs):
            response = super().call(stage, operation, **kwargs)
            return controlled_stop() if operation == "stage.stop" else response

    assert staged_benchmark.verify_stopped(controlled_stop()) == "trusted-controller-stop"
    for failure in (None, "raw-result"):
        result = staged_benchmark.benchmark(
            ControlledSession(failure), "base", CONFIG, tmp_path / str(failure))
        assert result["status"] == ("passed" if failure is None else "failed")
        assert result["stop_classification"] == "trusted-controller-stop"
        assert result["completion_requests"] == 1


@pytest.mark.parametrize("case", ["native-exit", "STOP", "timeout", "unclean", "active", "disconnected", "quarantined"])
def test_controlled_stop_rejects_abnormal_or_unconfirmed_exit(case):
    value = controlled_stop()
    if case == "native-exit":
        value["receipt"].pop("error")
    elif case in ("STOP", "timeout"):
        value["returncode"] = 0  # Even zero must not override the watch failure.
        value["receipt"]["error"] = ("InterruptedError: device STOP exists" if case == "STOP"
                                     else "TimeoutError: stage exceeded finite deadline")
    elif case == "unclean":
        value["receipt"]["guardian"]["cleanup"]["processes_gone"] = False
    elif case == "active":
        value["receipt"]["guardian"]["cleanup"]["state"]["ActiveState"] = "active"
    elif case == "disconnected":
        value["receipt"]["guardian"]["controller_disconnected"] = False
    else:
        value["state"] = "quarantined"
    with pytest.raises(ValueError, match="did not stop normally and cleanly"):
        staged_benchmark.verify_stopped(value)
