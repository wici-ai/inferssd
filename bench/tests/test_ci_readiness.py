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


@pytest.mark.parametrize("backend", ["CUDA0", "Metal"])
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
