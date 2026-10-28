from __future__ import annotations

import copy
import gzip
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_log_records import REFERENCE, REFERENCE_ONLINE, exact
from test_offline_planner import DATA as PLAN_DATA
from test_settings_command import DATA, CommandTranscript, runtime_key

from inferssd import cli
from inferssd.plan import settings_recorded
from inferssd.telemetry import legacy, recorded
from inferssd.telemetry.files import LOCAL_FILES, readable


@pytest.mark.parametrize("compressed", [False, True])
def test_real_recorded_files_and_encoding(tmp_path, compressed):
    data = b"irrelevant \xff\n[kimi_host_ledger] reason=destroy margin=12 unified=0\n"
    if compressed:
        (tmp_path / "server.log.gz").write_bytes(gzip.compress(data, mtime=0))
        (tmp_path / "server.log").write_text("[kimi_host_ledger] reason=destroy margin=99\n")
    else:
        (tmp_path / "server.log").write_bytes(data)
    (tmp_path / "client.jsonl").write_text('{"tok_s":[0,null,1]}\n')
    (tmp_path / "meta.txt").write_text("load_s=1.25")
    assert exact(recorded.read_arm(str(tmp_path))) == exact(REFERENCE.read_arm(str(tmp_path)))
    assert recorded.ledger(str(tmp_path)) == REFERENCE.ledger(str(tmp_path))
    (tmp_path / "client.jsonl").write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        recorded.read_arm(str(tmp_path))
    (tmp_path / "client.jsonl").unlink()
    (tmp_path / "meta.txt").write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        recorded.read_arm(str(tmp_path))


def test_invalid_gzip_is_not_a_fallback_request(tmp_path):
    (tmp_path / "server.log.gz").write_bytes(b"invalid compressed data")
    (tmp_path / "server.log").write_text("[kimi_host_ledger] reason=destroy margin=99\n")
    for function in (recorded.read_arm, recorded.ledger):
        with pytest.raises(gzip.BadGzipFile):
            function(str(tmp_path))


def test_readability_is_open_based(tmp_path):
    assert not readable(str(tmp_path), LOCAL_FILES)
    assert not readable(str(tmp_path / "absent"), LOCAL_FILES)
    path = tmp_path / "record"
    path.write_text("x")
    with patch("builtins.open", side_effect=PermissionError("synthetic unreadable file")):
        assert not readable(str(path), LOCAL_FILES)


def test_supplied_parser_bypasses_log_but_reloads_client(tmp_path):
    (tmp_path / "server.log.gz").write_bytes(b"invalid compressed data")
    parser = legacy.ArmParser()
    assert recorded.read_arm(str(tmp_path), parser=parser)["client"] == []
    (tmp_path / "client.jsonl").write_text('{"turn":1}\n')
    assert recorded.read_arm(str(tmp_path), parser=parser)["client"] == [{"turn": 1}]


def test_original_live_complete_line_consumer_accepts_new_parser(tmp_path):
    # This is the frozen reader on synthetic files, not an online controller.
    path = tmp_path / "server.log"
    records = [
        '[kimi_cpu_sink_trace] {"phase":"decode","event":"layer",'
        f'"epoch":1,"token":{n},"layer":0,"selected_experts":[0],'
        '"vram_hit":[false],"cache_hit":[false]}\n'
        for n in (1, 2, 1)
    ]

    def run(parser, read_arm):
        snapshots = []
        with (
            patch.object(REFERENCE_ONLINE, "ArmParser", parser),
            patch.object(REFERENCE_ONLINE, "read_arm", read_arm),
            patch.object(REFERENCE_ONLINE, "_LIVE", {}),
            patch.object(REFERENCE_ONLINE, "_ARMS", {}),
        ):
            path.write_bytes(b"")
            for data in (records[0][:40], records[0][40:] + records[1], records[2]):
                with path.open("ab") as stream:
                    stream.write(data.encode())
                result = REFERENCE_ONLINE._read_arm_cached(str(tmp_path), 0)
                snapshots.append(exact(result))
                snapshots.append(REFERENCE_ONLINE._LIVE[str(path)][1])
        return snapshots

    assert run(legacy.ArmParser, recorded.read_arm) == run(REFERENCE.ArmParser, REFERENCE.read_arm)


def populate(root, files):
    for name, value in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value if isinstance(value, str) else json.dumps(value))


@pytest.mark.parametrize("name", ["plan-choice", "unified-plan", "derived-pending-empty"])
def test_actual_settings_cli(tmp_path, capsys, name):
    case = next(c for c in DATA["cases"] if c["name"] == name)
    populate(tmp_path, case["files"])
    output = tmp_path / "settings.env"
    with patch("os.cpu_count", side_effect=AssertionError("must not query current host")):
        assert cli.main(["settings", str(tmp_path), "-o", str(output)]) == 0
    expected = case["expected"][runtime_key()]["writes"]["/case/settings.env"]
    assert output.read_text() == expected.replace("/case/run", str(tmp_path))
    assert not Path(f"{output}.tmp").exists()
    assert capsys.readouterr().out == ""


def test_actual_plan_reload_settings_cli_chain(tmp_path, capsys):
    case = next(c for c in PLAN_DATA["cases"] if c["name"] == "complete-discrete")
    populate(tmp_path, case["files"])
    ticks = iter(case["clock"] if "clock" in case else [100.0, 101.25])
    with patch.object(cli, "time", SimpleNamespace(time=lambda: next(ticks))):
        assert (
            cli.main(["plan", str(tmp_path), *case["argv"], "-o", str(tmp_path / "plan.json")]) == 0
        )
    assert cli.main(["settings", str(tmp_path), "-o", str(tmp_path / "settings.env")]) == 0
    expected = case["expected"][runtime_key()]
    assert (tmp_path / "plan.json").read_text() == expected["writes"]["plan.json"]
    assert (tmp_path / "settings.env").read_text() == expected["writes"][
        "/case/settings.env"
    ].replace("/case/run", str(tmp_path))
    assert capsys.readouterr().out == expected["stdout"]


def test_settings_cli_lazy_explicit_host(tmp_path):
    case = copy.deepcopy(next(c for c in DATA["cases"] if c["name"] == "host-needed"))
    populate(tmp_path, case["files"])
    output = tmp_path / "settings.env"
    with patch("os.cpu_count", side_effect=AssertionError("must not query current host")):
        with pytest.raises(ValueError, match="explicit --host-threads"):
            cli.main(["settings", str(tmp_path), "--cpus", "split", "-o", str(output)])
        assert not output.exists()
        assert (
            cli.main(
                [
                    "settings",
                    str(tmp_path),
                    "--cpus",
                    "split",
                    "--host-threads",
                    "8",
                    "-o",
                    str(output),
                ]
            )
            == 0
        )


@pytest.mark.parametrize("fault", [None, "fail_open", "fail_write", "fail_close", "fail_replace"])
def test_real_output_adapter_fault_order(fault):
    case = {"files": {}, **({fault: True} if fault else {})}
    tr = CommandTranscript(case)
    with (
        patch.object(settings_recorded, "open", tr.open, create=True),
        patch.object(settings_recorded, "os", SimpleNamespace(replace=tr.replace)),
    ):
        if fault:
            with pytest.raises(OSError):
                settings_recorded.write_settings("/case/settings.env", "complete text")
        else:
            settings_recorded.write_settings("/case/settings.env", "complete text")
    if fault == "fail_open":
        assert tr.writes == {}
    elif fault:
        assert "/case/settings.env.tmp" in tr.writes
        assert "/case/settings.env" not in tr.writes
    else:
        assert tr.writes == {"/case/settings.env": "complete text"}
    if fault in ("fail_open", "fail_write", "fail_close"):
        assert not any(event[0] == "replace" for event in tr.events)


def test_imports_have_no_file_host_or_process_effects():
    script = """
import builtins, os, pathlib, socket, subprocess, sys
before = sys.path[:]
def denied(*args, **kwargs):
    raise AssertionError('import attempted external work')
builtins.open = denied
pathlib.Path.open = denied
os.open = denied
os.cpu_count = denied
socket.socket = denied
subprocess.Popen = denied
import inferssd.cli
from inferssd.telemetry.legacy import ArmParser
from inferssd.telemetry.files import LocalArmFiles
ArmParser()
LocalArmFiles()
assert sys.path == before
assert 'logparse' not in sys.modules and 'controller' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
