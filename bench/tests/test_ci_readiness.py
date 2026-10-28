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


def mac_cache_events():
    from inferssd.configuration.settings import parse

    settings = parse('inferssd-settings 1\nhost.expert_cache_mib = 1\n', platform="darwin").document()
    return [dict(v=1, seq=0, ev="ssd.settings", **settings),
            dict(v=1, seq=1, ev="ssd.controller.begin", schema=1),
            dict(v=1, seq=2, ev="ssd.mac.cache", reason="periodic", requests=1024, hits=256,
                 misses=768, cache_bytes=512, budget_bytes=1 << 20, read_bytes=4096, peak_staging_bytes=128),
            dict(v=1, seq=3, ev="ssd.mac.cache", reason="shutdown", requests=1536, hits=512,
                 misses=1024, cache_bytes=1024, budget_bytes=1 << 20, read_bytes=8192, peak_staging_bytes=256)]


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_native_observation_stage_uses_the_real_consumer(tmp_path, monkeypatch, platform):
    import json
    import subprocess
    from ci import staged_observations as observations

    values = [
        dict(v=1, seq=0, ev="ssd.controller.begin", schema=1),
        dict(v=1, seq=1, ev="ssd.place.layer", epoch=1, token=32, layer=0,
             decode=True, success=True, has_token=True, selected_experts=[7],
             vram_hit=[False], cache_hit=[True], wall_us=9, wait_io_us=2, compute_us=5),
    ]
    if platform == "darwin":
        values = mac_cache_events()
    raw = "".join(json.dumps(value) + "\n" for value in values).encode()
    source = tmp_path / "native events.jsonl"
    source.write_bytes(raw)
    path = str(source) if platform == "darwin" else "/scratch/run/native-events.jsonl"

    class Session:
        def start(self, name, kind, argv, environment, cwd, timeout):
            assert (name, kind, environment, cwd, timeout) == (
                "candidate-observations-muse", "test", {}, "/scratch", 30)
            if platform == "linux":
                assert argv == ["/usr/bin/cat", path]
            else:
                assert argv[:3] == ["/usr/bin/python3", "-I", "-c"]
                assert argv[-1] == path and len(argv) == 5
                # Exercise the reader itself, including an argument with spaces;
                # the real Mac broker selects its installed Python interpreter.
                assert subprocess.check_output([sys.executable, *argv[1:]]) == raw
            return "stage"

        def wait(self, stage, *, deadline):
            assert stage == "stage" and deadline > 0
            return dict(returncode=0, cleanup_confirmed=True)

    def collect_log(session, stage, output):
        (output / "stdout.log").write_bytes(raw)

    monkeypatch.setattr(observations, "collect_log", collect_log)
    output = tmp_path / "observation"
    receipt = observations.observe(Session(), dict(path=path, cwd="/scratch"), "muse", output,
                                   platform=platform)
    assert receipt["status"] == "passed"
    if platform == "linux":
        assert (receipt["groups"], receipt["trace"], receipt["prof"]) == (1, 1, 1)
    else:
        assert receipt["observation_kind"] == "mac-cache" and receipt["cache_samples"] == 2
        assert (receipt["requests"], receipt["hits"], receipt["misses"], receipt["read_bytes"]) == (1536, 512, 1024, 8192)
        assert not {"groups", "trace", "prof"} & receipt.keys()
    assert receipt["legacy_log_present"] is False
    assert (output / "native-events.jsonl").read_bytes() == raw
    assert json.loads((output / "observation.json").read_text()) == receipt


@pytest.mark.parametrize("damage", ["sequence", "settings", "no-cache", "no-shutdown", "early-shutdown",
                                    "accounting", "budget", "over-budget", "decreased-requests",
                                    "decreased-hits", "decreased-misses", "decreased-reads", "no-requests", "no-reads"])
def test_mac_cache_observations_reject_incomplete_or_inconsistent_events(tmp_path, damage):
    import json
    from ci.staged_observations import mac_cache_observations

    values = mac_cache_events()
    final = values[-1]
    if damage == "sequence": final["seq"] += 1
    elif damage == "settings": values[0]["values"].pop("host.expert_cache_mib")
    elif damage == "no-cache": values = values[:2]
    elif damage == "no-shutdown": final["reason"] = "periodic"
    elif damage == "early-shutdown": values[-2]["reason"] = "shutdown"
    elif damage == "accounting": final["requests"] += 1
    elif damage == "budget": final["budget_bytes"] += 1
    elif damage == "over-budget": final["cache_bytes"] = final["budget_bytes"] + 1
    elif damage == "decreased-requests": final.update(requests=512, hits=256, misses=256)
    elif damage == "decreased-hits": final.update(hits=255, misses=1281)
    elif damage == "decreased-misses": final.update(hits=769, misses=767)
    elif damage == "decreased-reads": final["read_bytes"] = 4095
    elif damage == "no-requests":
        for row in values[2:]: row.update(requests=0, hits=0, misses=0)
    elif damage == "no-reads":
        for row in values[2:]: row["read_bytes"] = 0
    path = tmp_path / "native-events.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in values))
    with pytest.raises(ValueError): mac_cache_observations(path)


@pytest.mark.parametrize("field", ["requests", "hits", "misses", "cache_bytes", "budget_bytes", "read_bytes", "peak_staging_bytes"])
@pytest.mark.parametrize("invalid", [-1, True])
def test_mac_cache_counters_require_nonnegative_integers(tmp_path, field, invalid):
    import json
    from ci.staged_observations import mac_cache_observations

    values = mac_cache_events()
    values[-1][field] = invalid
    path = tmp_path / "native-events.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in values))
    with pytest.raises(ValueError, match="nonnegative integers"): mac_cache_observations(path)


def test_smoke_waits_through_health_timeout_and_sends_one_completion(tmp_path, monkeypatch):
    import base64
    from dataclasses import asdict
    import json
    from ci import staged_smoke as smoke
    from ci.stage_session import Stage

    class Session:
        health = 0
        completions = 0
        probes = []

        def start(self, *args):
            return Stage('smoke', 'handle', 41, 62002, {'birth': 9}, 'owned-scope')

        def call(self, stage, operation, **kwargs):
            if operation == 'stage.poll':
                return dict(running=True)
            if operation == 'stage.probe':
                self.probes.append(kwargs.get('light', False))
                return dict(asdict(stage), ready=True)
            if operation == 'stage.stop':
                return dict(running=False, returncode=0, cleanup_confirmed=True)
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
    assert session.probes == [True, True, True, False, False]
    assert [json.loads(p.read_text())['status'] for p in sorted(
        (tmp_path / 'run').glob('health-attempt-*.json'))] == [0, 503, 200]


@pytest.mark.parametrize('failure_point', ['headers', 'body'])
@pytest.mark.parametrize('socket_timeout_is_alias', [True, False])
def test_local_http_timeout_is_recorded_and_never_accepted_as_completion(
        monkeypatch, failure_point, socket_timeout_is_alias):
    from ci import staged_chat as chat

    class LegacySocketTimeout(OSError):
        pass

    timeout_type = TimeoutError if socket_timeout_is_alias else LegacySocketTimeout
    monkeypatch.setattr(chat.socket, 'timeout', timeout_type)

    class Connection:
        closed = False
        status = 200

        def request(self, *args):
            pass

        def getresponse(self):
            if failure_point == 'headers':
                raise timeout_type('headers timed out')
            return self

        def readline(self, *args):
            raise timeout_type('body timed out')

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
