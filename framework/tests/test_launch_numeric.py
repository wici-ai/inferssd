"""Explicit typed-input tightening; original F observations remain unchanged."""

import hashlib
import json
from pathlib import Path

import pytest
from test_launch_linux import replay

FIXTURES = Path(__file__).parent / "fixtures"
NUMERIC = json.loads((FIXTURES / "launch-linux-numeric-F.json").read_text())


def test_numeric_frozen_bytes():
    m = json.loads((FIXTURES / "launch-numeric-F-manifest.json").read_text())
    assert (
        hashlib.sha256((FIXTURES / "launch-linux-numeric-F.json").read_bytes()).hexdigest()
        == m["golden_sha256"]
    )
    assert (
        hashlib.sha256((Path(__file__).parent / "launch_numeric_cases.py").read_bytes()).hexdigest()
        == m["cases_sha256"]
    )


@pytest.mark.parametrize("record", NUMERIC, ids=lambda r: r["case"]["name"])
def test_numeric_input_boundary(record):
    old = record["runs"][0]
    actual, _ = replay(record, old)
    stdout = "".join(e.text for e in actual.outputs if e.channel == "stdout")
    stderr = "".join(e.text for e in actual.outputs if e.channel == "stderr")
    if record["case"]["name"] == "checkpoint-leading-zero":
        assert actual.code == old["code"] == 0
        assert stdout == old["stdout"] and stderr == old["stderr"]
        assert actual.argv[actual.argv.index("--ctx-checkpoints") + 1] == "0008"
    else:
        key = next(k for k in record["case"]["env"] if k != "MEMORY_MAX")
        assert actual.code == 2
        assert key + "=" in stderr
        assert "from environment: expected unsigned decimal integer data" in stderr
        assert actual.argv == ()


def test_numeric_error_identifies_settings_source():
    import copy

    from test_launch_linux import GOLDENS

    record = copy.deepcopy(GOLDENS[0])
    record["case"]["settings"] = "VRAM_RESERVE_MIB=010\n"
    record["env"]["SETTINGS"] = "@case/settings.env"
    actual, _ = replay(record, record["runs"][0])
    assert "from SETTINGS @case/settings.env" in "".join(e.text for e in actual.outputs)
