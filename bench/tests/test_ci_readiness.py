"""Readiness orchestration contracts, not hardware or performance evidence."""
import csv
import io
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.staged_numerical import validate_csv
from ci.staged_integration import DEVICES


def records(backend="CUDA0", supported="1", error=""):
    out = io.StringIO()
    writer = csv.DictWriter(out, ["backend_name", "op_name", "test_mode", "supported", "error_message"])
    writer.writeheader()
    writer.writerow(dict(backend_name=backend, op_name="ADD", test_mode="test", supported=supported, error_message=error))
    return out.getvalue().encode()


def test_readiness_devices_exclude_local_performance_hosts():
    assert DEVICES == {"cloud-5090": "linux", "dgx-spark": "linux", "local-mbp": "darwin"}


@pytest.mark.parametrize("backend", ["CUDA0", "MTL0"])
def test_actual_requested_gpu_rows_are_required(backend):
    assert validate_csv(records(backend), backend) == 1
    for raw in (records("CPU"), records(backend, "0"), b"backend_name,supported\n"):
        with pytest.raises(ValueError, match="no supported tests"):
            validate_csv(raw, backend)
    with pytest.raises(ValueError, match="comparison failed"):
        validate_csv(records(backend, error="compare failed"), backend)


def test_negative_finish_receipt_never_reports_ready(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ci import staged_integration as integration
    import json
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(dict(
        revision='a' * 40, device='cloud-5090', paths={'platform': 'linux'},
        smoke={'model': 'muse'}, numerical={'backend': 'CUDA0'}, manifest_sha256='b' * 64)))
    class NegativeCleanup:
        closed = False
        def __init__(self, *args, **kwargs): pass
        def open(self): raise RuntimeError('model not started')
        def finish(self): return {'cleanup_confirmed': False}
        def close(self): self.closed = True
    monkeypatch.setattr(integration, 'StageSession', NegativeCleanup)
    result = tmp_path / 'result'
    args = SimpleNamespace(config=config, config_sha256=integration.digest(config),
                           output=result, socket='/unused', job_id='c' * 32)
    assert integration.run(args) == 1
    report = json.loads((result / 'integration.json').read_text())
    assert report['status'] == 'failed' and report['cleanup'] == 'failed'
    assert report['cleanup_confirmed'] is False
    assert report['functional'] == 'failed' and report['performance'] == 'not-measured'
    assert (result / 'evidence-manifest.json').is_file()


def test_native_observation_stage_uses_the_real_consumer(tmp_path, monkeypatch):
    import json
    from ci import staged_observations as observations

    values = [
        dict(v=1, seq=0, ev="ssd.controller.begin", schema=1),
        dict(v=1, seq=1, ev="ssd.place.layer", epoch=1, token=32, layer=0,
             decode=True, success=True, has_token=True, selected_experts=[7],
             vram_hit=[False], cache_hit=[True], wall_us=9, wait_io_us=2, compute_us=5),
    ]
    raw = "".join(json.dumps(value) + "\n" for value in values).encode()

    class Session:
        def start(self, name, kind, argv, environment, cwd, timeout):
            assert (name, kind, argv, environment, cwd, timeout) == (
                "candidate-observations-muse", "test",
                ["/usr/bin/cat", "/scratch/run/native-events.jsonl"], {}, "/scratch", 30)
            return "stage"

        def wait(self, stage, *, deadline):
            assert stage == "stage" and deadline > 0
            return dict(returncode=0, cleanup_confirmed=True)

    def collect_log(session, stage, output):
        (output / "stdout.log").write_bytes(raw)

    monkeypatch.setattr(observations, "collect_log", collect_log)
    output = tmp_path / "observation"
    receipt = observations.observe(Session(), dict(path="/scratch/run/native-events.jsonl",
                                                  cwd="/scratch"), "muse", output)
    assert receipt["status"] == "passed"
    assert (receipt["groups"], receipt["trace"], receipt["prof"]) == (1, 1, 1)
    assert receipt["legacy_log_present"] is False
    assert (output / "native-events.jsonl").read_bytes() == raw
    assert json.loads((output / "observation.json").read_text()) == receipt
