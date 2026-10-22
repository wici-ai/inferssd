from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from ssdash.contracts import ContractError, DeviceProfile
from ssdash.probe.device_profile import aggregate
from ssdash.probe.recorded import load_reports

FIXTURES = Path(__file__).parent / "fixtures"
GOLDENS = json.loads((FIXTURES / "device-goldens.json").read_text())
CASES = GOLDENS["cases"]


def materialize(case, root):
    probe, experts = root / case.get("probe_dir_name", "probe"), root / "experts"
    probe.mkdir()
    experts.mkdir()
    for name, text in case["probe_files"].items():
        (probe / name).write_text(text)
    for name, value in case.get("probe_binary_hex", {}).items():
        (probe / name).write_bytes(bytes.fromhex(value))
    (experts / "experts.json").write_text(json.dumps(case["experts"]))
    if "experts_binary_hex" in case:
        (experts / "experts.json").write_bytes(bytes.fromhex(case["experts_binary_hex"]))
    return probe, experts


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_f_oracle_is_frozen(case, tmp_path):
    oracle = FIXTURES / "oracle/device_profile_f.py"
    manifest = json.loads((FIXTURES / "oracle/manifest.json").read_text())
    assert hashlib.sha256(oracle.read_bytes()).hexdigest() == manifest["sha256"]
    assert manifest["sha256"] == GOLDENS["oracle_sha256"]
    probe, experts = materialize(case, tmp_path)
    run = subprocess.run(
        [sys.executable, str(oracle), str(probe), str(experts)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == case["legacy"]["returncode"]
    assert run.stdout == case["legacy"]["stdout"]
    error = run.stderr.splitlines()[-1].split(":", 1)[0] if run.stderr else None
    assert error == case["legacy"]["error"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_extracted_profile_matches_f_or_declared_boundary(case, tmp_path):
    probe, experts = materialize(case, tmp_path)
    if "new_error" in case:
        errors = {
            "KeyError": KeyError,
            "ValueError": ValueError,
            "IndexError": IndexError,
            "TypeError": TypeError,
            "UnicodeDecodeError": UnicodeDecodeError,
            "ContractError": ContractError,
        }
        with pytest.raises(errors[case["new_error"]]):
            aggregate(*load_reports(probe, experts))
    else:
        profile = aggregate(*load_reports(probe, experts))
        assert profile.to_json() == case["legacy"]["stdout"]
        assert DeviceProfile.from_json(profile.to_json()) == profile


def test_result_is_deeply_immutable_and_serialization_does_not_alias():
    profile = DeviceProfile.from_json(CASES[0]["legacy"]["stdout"])
    original = profile.to_json()
    with pytest.raises(FrozenInstanceError):
        profile.mem_gib.total = 0
    with pytest.raises(TypeError):
        profile.cpus[0] = profile.cpus[1]
    with pytest.raises(TypeError):
        profile.ssd[0].gibs_runs[0] = 0
    with pytest.raises(FrozenInstanceError):
        profile.lanes.fields[0][1].fixed_us = 999
    rendered = profile.to_mapping()
    rendered["cpus"]["0"]["capacity"] = 0
    rendered["lanes"]["cpu"]["fixed_us"] = 0
    assert profile.to_json() == original


@pytest.mark.parametrize("schema", [0, 1, 3, 2.0, "2", True, None])
def test_unknown_or_untyped_schema_rejected(schema):
    data = json.loads(CASES[0]["legacy"]["stdout"])
    data["schema"] = schema
    with pytest.raises(ContractError, match="unsupported device schema"):
        DeviceProfile.from_mapping(data)


def test_aggregate_does_not_consult_current_machine(tmp_path, monkeypatch):
    probe, experts = materialize(CASES[0], tmp_path)
    reports, metadata = load_reports(probe, experts)

    def denied(*args, **kwargs):
        raise AssertionError("pure profile consulted an external dependency")

    monkeypatch.setattr(Path, "read_text", denied)
    monkeypatch.setattr("builtins.open", denied)
    monkeypatch.setattr("os.cpu_count", denied)
    monkeypatch.setattr("os.getenv", denied)
    monkeypatch.setattr(subprocess, "Popen", denied)
    assert aggregate(reports, metadata).to_json() == CASES[0]["legacy"]["stdout"]


def test_loader_dependencies_are_explicit_and_permission_errors_not_missing():
    case = CASES[0]
    records = {Path("reports") / key: value for key, value in case["probe_files"].items()}
    records[Path("experts/experts.json")] = json.dumps(case["experts"])

    def read(path):
        if path not in records:
            raise FileNotFoundError(path)
        return records[path]

    def find(directory, pattern):
        return [path for path in records if path.parent == directory and path.match(pattern)]

    reports, metadata = load_reports(Path("reports"), Path("experts"), read_text=read, find=find)
    assert aggregate(reports, metadata).to_json() == case["legacy"]["stdout"]

    def denied(path):
        if path.name == "files.txt":
            raise PermissionError(path)
        return read(path)

    with pytest.raises(PermissionError):
        load_reports(Path("reports"), Path("experts"), read_text=denied, find=find)


def test_imports_have_no_cli_file_or_process_side_effects():
    program = """
import argparse, dataclasses, json, math, pathlib, re, typing, subprocess, os, sys
def denied(*a, **k):
    raise AssertionError("import performed external work")
pathlib.Path.read_text = denied
pathlib.Path.glob = denied
subprocess.Popen = denied
os.cpu_count = denied
os.getenv = denied
sys.argv = ["unrelated", "/does-not-exist"]
import ssdash, ssdash.cli, ssdash.contracts
import ssdash.probe.device_profile, ssdash.probe.recorded
"""
    subprocess.run([sys.executable, "-c", program], check=True)


def test_cli_only_aggregates_supplied_reports(tmp_path):
    probe, experts = materialize(CASES[0], tmp_path)
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "ssdash",
            "probe",
            "--from-reports",
            str(probe),
            "--experts",
            str(experts),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert run.stdout == CASES[0]["legacy"]["stdout"]
    profile = tmp_path / "device.json"
    profile.write_text(run.stdout)
    result = subprocess.run(
        [sys.executable, "-m", "ssdash", "validate", "--device", str(profile)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == "device schema 2: valid\n"


def test_cli_rejects_unknown_schema_without_a_traceback(tmp_path):
    path = tmp_path / "device.json"
    path.write_text('{"schema": 100}')
    run = subprocess.run(
        [sys.executable, "-m", "ssdash", "validate", "--device", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 2
    assert "unsupported device schema" in run.stderr
    assert "Traceback" not in run.stderr


def test_numeric_overflow_is_a_contract_error_and_cli_diagnostic(tmp_path):
    data = json.loads(CASES[0]["legacy"]["stdout"])
    data["cpus"]["0"]["capacity"] = 10**1000
    with pytest.raises(ContractError, match="finite numeric range"):
        DeviceProfile.from_mapping(data)
    path = tmp_path / "device.json"
    path.write_text(json.dumps(data))
    run = subprocess.run(
        [sys.executable, "-m", "ssdash", "validate", "--device", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 2
    assert "finite numeric range" in run.stderr
    assert "Traceback" not in run.stderr


def test_valid_primary_never_reads_fallback():
    case = CASES[0]
    records = {Path("reports") / key: value for key, value in case["probe_files"].items()}
    records[Path("experts/experts.json")] = json.dumps(case["experts"])

    def read(path):
        if path.name in ("fio-file0.json", "fio-file1.json"):
            raise AssertionError("valid primary must short-circuit fallback reads")
        if path not in records:
            raise FileNotFoundError(path)
        return records[path]

    def find(directory, pattern):
        return [path for path in records if path.parent == directory and path.match(pattern)]

    result = aggregate(*load_reports(Path("reports"), Path("experts"), read_text=read, find=find))
    assert result.to_json() == case["legacy"]["stdout"]
