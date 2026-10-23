"""Actual typed command/CLI comparisons against independently frozen whole F."""

import gzip
import inspect
import json
import subprocess
import sys

import pytest
from bootstrap_record_cases import CASES
from test_bootstrap_records import FIXTURES, observe, runtime_key

from inferssd.bootstrap.core_sets import parse_list, run_coresets
from inferssd.bootstrap.tier_slots import reached, run_ubprobe


def execute_candidate(case, host):
    if case["kind"] == "parse_list":
        return parse_list(case["value"])
    if case["kind"] == "reached":
        return list(reached(iter(case["lines"])))
    if case["kind"] == "coresets":
        return run_coresets(case["argv"], open_file=host.open, emit=host.emit)
    return run_ubprobe(case["argv"], open_file=host.open, open_gzip=host.gzip_open, emit=host.emit)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_candidate_whole_F_transcript(case):
    expected = json.loads((FIXTURES / "bootstrap-records-goldens.json").read_text())
    assert observe(case, execute_candidate) == expected[runtime_key()][case["name"]]


@pytest.mark.parametrize("function", [run_coresets, run_ubprobe])
def test_effect_ports_have_no_runtime_defaults(function):
    for parameter in inspect.signature(function).parameters.values():
        assert parameter.default is inspect.Parameter.empty


def test_real_offline_coresets_cli(tmp_path):
    case = next(c for c in CASES if c["name"] == "groups-unions-reserve-one")
    path = tmp_path / "cpus.txt"
    path.write_text(case["files"]["cpus.txt"])
    result = subprocess.run(
        [sys.executable, "-m", "inferssd", "bootstrap-records", "coresets", str(path)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout == observe(case)["stdout"]
    assert result.stderr == ""


def test_real_offline_gzip_cli(tmp_path):
    case = next(c for c in CASES if c["name"] == "ordered-duplicate-ubs")
    pipeline = tmp_path / "recorded"
    pipeline.mkdir()
    for ub in ("32", "16"):
        run = tmp_path / "runs" / ub
        run.mkdir(parents=True)
        (pipeline / f"ubprobe-{ub}.out").write_text(str(run) + "\nignored\n")
        with gzip.open(run / "server.log.gz", "wt") as stream:
            stream.write(case["files"][f"runs/{ub}/server.log.gz"])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "inferssd",
            "bootstrap-records",
            "tier-slots",
            str(pipeline),
            "32",
            "16",
            "32",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout == observe(case)["stdout"]
    assert result.stderr == ""
