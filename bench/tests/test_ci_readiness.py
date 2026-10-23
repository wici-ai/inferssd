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


def test_smoke_waits_through_health_timeout_and_sends_one_completion(tmp_path, monkeypatch):
    import base64
    from dataclasses import asdict
    import json
    from ci import staged_smoke as smoke
    from ci.stage_session import Stage

    class Session:
        health = 0
        completions = 0

        def start(self, *args):
            return Stage('smoke', 'handle', 41, 62002, {'birth': 9}, 'owned-scope')

        def call(self, stage, operation, **kwargs):
            if operation == 'stage.poll':
                return dict(running=True)
            if operation == 'stage.probe':
                return dict(asdict(stage), ready=True)
            if operation == 'stage.stop':
                return dict(returncode=0, cleanup_confirmed=True)
            assert operation == 'stage.client'
            request = kwargs['request']
            path, body, status = request['path'], {}, 200
            if path == '/health':
                self.health += 1
                if self.health == 1:
                    return dict(status=0, chunks=[], transport_error='TimeoutError: timed out',
                                started_monotonic_ns=100, ended_monotonic_ns=200)
                status = 503 if self.health == 2 else 200
            elif path == '/apply-template':
                body = dict(tokens=[1, 2])
            elif path == '/props':
                body = dict(total_slots=1, default_generation_settings=dict(n_ctx=1024))
            else:
                assert path == '/completion' and request['body']['n_predict'] == 8
                self.completions += 1
                body = dict(tokens_predicted=8, truncated=False, stop_type='limit')
            return dict(status=status, started_monotonic_ns=100, ended_monotonic_ns=200,
                        chunks=[dict(elapsed_ns=1,
                                     data_b64=base64.b64encode(json.dumps(body).encode()).decode())])

        def output(self, *args):
            return b'', 0, True

    monkeypatch.setattr(smoke.time, 'sleep', lambda _: None)
    monkeypatch.setattr(smoke, 'host_memory', lambda *_: None)
    config = dict(model='muse', revision='a' * 40, argv=['/sealed/server'],
                  environment={}, cwd='/scratch', port=18080, timeout_s=30,
                  request_timeout_s=10)
    session = Session()
    result = smoke.smoke(session, config, tmp_path / 'run')
    assert result['status'] == 'passed' and result['cleanup_confirmed'] is True
    assert session.completions == 1 and session.health == 3
    assert [json.loads(p.read_text())['status'] for p in sorted(
        (tmp_path / 'run').glob('health-attempt-*.json'))] == [0, 503, 200]


@pytest.mark.parametrize('failure_point', ['headers', 'body'])
def test_local_http_timeout_is_recorded_and_never_accepted_as_completion(monkeypatch, failure_point):
    from ci import staged_chat as chat

    class Connection:
        closed = False
        status = 200

        def request(self, *args):
            pass

        def getresponse(self):
            if failure_point == 'headers':
                raise TimeoutError('headers timed out')
            return self

        def readline(self, *args):
            raise TimeoutError('body timed out')

        def close(self):
            self.closed = True

    connection = Connection()
    monkeypatch.setattr(chat.http.client, 'HTTPConnection', lambda *a, **k: connection)
    receipt = chat.local_request(18080, 'GET', '/health', None, 2)
    assert connection.closed and receipt['status'] == 0
    assert receipt['transport_error'].startswith('TimeoutError:')
    chat.validate_response(receipt, (0,))
    with pytest.raises(ValueError, match='did not succeed'):
        chat.validate_response(receipt)
