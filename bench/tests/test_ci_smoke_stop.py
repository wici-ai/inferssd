"""Plain smoke retains the real broker stop evidence; no native/device claim."""

import copy
from dataclasses import asdict
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci import staged_smoke
from ci.contract import load
from ci.stage_session import Stage
from test_ci_staged_benchmark import controlled_stop
from test_ci_selection_lifetime import mac_stop


def run_smoke(tmp_path, monkeypatch, stopped, *, valid_completion=True):
    class Session:
        completions = 0

        def start(self, *args):
            return Stage("smoke", "handle", 41, 62002, {"birth": 9},
                         stopped.get("scope_identity", "owned-scope"))

        def call(self, stage, operation, **kwargs):
            if operation == "stage.poll":
                return dict(running=True)
            if operation == "stage.probe":
                return dict(asdict(stage), running=True, ready=True)
            assert operation == "stage.stop"
            return copy.deepcopy(stopped)

        def output(self, *args):
            return b"native log\n", 11, True

    class HTTP:
        def __init__(self, session, *args): self.session = session
        def request(self, *args, **kwargs): return dict(status=200)
        def json(self, method, path, body, timeout):
            if path == "/apply-template": return dict(tokens=[1, 2])
            if path == "/props": return dict(total_slots=1, default_generation_settings=dict(n_ctx=1024))
            assert path == "/completion" and body["n_predict"] == 8
            self.session.completions += 1
            return dict(tokens_predicted=8 if valid_completion else 7,
                        truncated=False, stop_type="limit")

    monkeypatch.setattr(staged_smoke, "NativeHTTP", HTTP)
    monkeypatch.setattr(staged_smoke, "host_memory", lambda *args: None)
    session = Session()
    output = tmp_path / "smoke"
    result = staged_smoke.smoke(session, dict(model="muse", revision="a" * 40,
        argv=["/sealed/server"], environment={}, cwd="/scratch", port=18080,
        timeout_s=30, request_timeout_s=10), output)
    assert session.completions == 1
    assert load(output / "stopped.json") == stopped  # Never rewrite rc1 to rc0.
    return result


@pytest.mark.parametrize("stop,classification", [
    (dict(running=False, returncode=0, cleanup_confirmed=True), "normal-exit"),
    (controlled_stop(), "trusted-controller-stop"),
    (mac_stop(), "mac-requested-stop"),
])
def test_plain_smoke_uses_the_existing_strict_stop_contract(tmp_path, monkeypatch, stop, classification):
    result = run_smoke(tmp_path, monkeypatch, stop)
    assert result["status"] == "passed" and result["cleanup_confirmed"] is True
    assert result["stop_classification"] == classification


def test_controlled_cleanup_cannot_promote_failed_completion(tmp_path, monkeypatch):
    result = run_smoke(tmp_path, monkeypatch, controlled_stop(), valid_completion=False)
    assert result["status"] == "failed" and result["cleanup_confirmed"] is True
    assert result["stop_classification"] == "trusted-controller-stop"


@pytest.mark.parametrize("case", ["STOP", "deadline", "quarantine", "unclean",
                                  "uncontrolled-one", "native-two", "wrong-scope"])
def test_plain_smoke_rejects_abnormal_or_unconfirmed_stop(tmp_path, monkeypatch, case):
    stopped = controlled_stop()
    if case in ("STOP", "deadline"):
        stopped["returncode"] = 0
        stopped["receipt"]["error"] = ("InterruptedError: device STOP exists" if case == "STOP"
                                        else "TimeoutError: stage exceeded finite deadline")
    elif case == "quarantine": stopped["state"] = "quarantined"
    elif case == "unclean": stopped["cleanup_confirmed"] = False
    elif case == "uncontrolled-one": stopped["receipt"].pop("error")
    elif case == "native-two": stopped["returncode"] = 2
    else: stopped["scope_identity"] = "/system.slice/unrelated.service"
    result = run_smoke(tmp_path, monkeypatch, stopped)
    assert result["status"] == "failed" and "cleanup_error" in result
