from __future__ import annotations

import gzip
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_recorded_calibration import data, runtime_key, synthetic_files

from ssdash import cli


def write_arm(path, config, compressed=False):
    path.mkdir()
    for name, text in synthetic_files(config).items():
        if name == "server.log" and compressed:
            (path / (name + ".gz")).write_bytes(gzip.compress(text.encode(), mtime=0))
        else:
            (path / name).write_text(text)


def fit_args(root, *extra):
    return [
        "fit",
        "--run",
        f"all={root / 'arm'}",
        "--skip",
        "4",
        "--profile",
        str(root / "model.json"),
        "-o",
        str(root / "insitu.json"),
        *extra,
    ]


@pytest.mark.parametrize("compressed", [False, True])
def test_actual_fit_plan_reload_settings_bytes(tmp_path, capsys, compressed):
    case = next(c for c in data()["cases"] if c["name"] == "complete-plan-settings-chain")
    expected = case["expected"][runtime_key()]
    write_arm(tmp_path / "arm", {}, compressed)
    assert cli.main(fit_args(tmp_path)) == 0

    def normalize(text):
        return text.replace(str(tmp_path / "arm"), "/case/a").replace(str(tmp_path), "/case/run")

    assert normalize(capsys.readouterr().out) == expected["stdout"]
    for name in ("model.json", "insitu.json"):
        assert normalize((tmp_path / name).read_text()) == expected["writes"]["/case/" + name]
    (tmp_path / "probe").mkdir()
    (tmp_path / "probe/device.json").write_text(json.dumps(case["device"]))
    ticks = iter([100.0, 101.25])
    with patch.object(cli, "time", SimpleNamespace(time=lambda: next(ticks))):
        assert (
            cli.main(
                [
                    "plan",
                    str(tmp_path),
                    "--prompts",
                    "32,17",
                    "--decode",
                    "8",
                    "--window",
                    "1-8",
                    "--known",
                    "-o",
                    str(tmp_path / "plan.json"),
                ]
            )
            == 0
        )
    # The old whole-chain observer writes this output outside /case/run.
    assert cli.main(["settings", str(tmp_path), "-o", str(tmp_path / "settings.env")]) == 0
    assert normalize(capsys.readouterr().out) == expected["chain"]["stdout"]
    assert (
        normalize((tmp_path / "plan.json").read_text()) == expected["chain"]["writes"]["plan.json"]
    )
    env = (tmp_path / "settings.env").read_text()
    assert normalize(env) == expected["chain"]["writes"]["/case/settings.env"]


def test_actual_late_ubatch_error_preserves_first_output(tmp_path):
    write_arm(tmp_path / "arm", {})
    (tmp_path / "ub.json").write_text("{")
    with pytest.raises(json.JSONDecodeError):
        cli.main(fit_args(tmp_path, "--ubatch-probe", str(tmp_path / "ub.json")))
    assert json.loads((tmp_path / "model.json").read_text())["schema"] == 2
    assert not (tmp_path / "insitu.json").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_actual_graph_report_error_preserves_both_outputs(tmp_path, capsys):
    write_arm(tmp_path / "arm", {})
    write_arm(tmp_path / "graph", {"graph_read_us": 1})
    with pytest.raises(ZeroDivisionError):
        cli.main(fit_args(tmp_path, "--graph", str(tmp_path / "graph")))
    assert json.loads((tmp_path / "model.json").read_text())["schema"] == 2
    assert json.loads((tmp_path / "insitu.json").read_text())["graph"]["passes"][0]["read_s"] == 0
    assert "== the GPU graph's prefill:" in capsys.readouterr().out
    assert not list(tmp_path.glob("*.tmp"))


def test_cli_has_no_run_acquisition(tmp_path):
    with pytest.raises(FileNotFoundError):
        cli.main(fit_args(tmp_path))
    assert list(tmp_path.iterdir()) == []
