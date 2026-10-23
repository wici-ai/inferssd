"""Candidate call graph against independently frozen complete F observations."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import inspect
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import test_recorded_calibration
from test_loop_candidate import candidate_api as loop_api
from test_offline_planner import runtime_key
from test_request_artifacts import data, observe


def arguments(cls, argv):
    parser = argparse.ArgumentParser()
    for field in dataclasses.fields(cls):
        key = "--" + field.name.replace("_", "-")
        if field.default is dataclasses.MISSING:
            parser.add_argument(key, required=True, type=int if field.name == "port" else str)
        elif isinstance(field.default, bool):
            parser.add_argument(key, action="store_true")
        else:
            parser.add_argument(
                key, default=field.default, type=int if isinstance(field.default, int) else str
            )
    return cls(**vars(parser.parse_args(argv)))


@contextlib.contextmanager
def candidate_api(h):
    from inferssd.bootstrap import client, requests
    from inferssd.loop.conversation import ConversationArguments, run_conversation
    from inferssd.loop.report import run_report
    from inferssd.telemetry.files import lines

    with loop_api(h) as model:

        def turn(*args):
            return h.turn(
                lambda *a: client.turn(*a, transport=h.urlopen, clock_ns=h.time_ns), *args
            )

        def bootstrap(argv):
            previous = sys.getprofile()

            def profile(frame, event, arg):
                if event == "return" and frame.f_code == requests.run_bootstrap.__code__:
                    h.ns["boot_client"] = frame.f_locals.copy()

            try:
                sys.setprofile(profile)
                requests.run_bootstrap(
                    arguments(requests.BootstrapArguments, argv),
                    turn=turn,
                    post=lambda *a: requests.post(*a, transport=h.urlopen),
                    open_file=h.open,
                    clock=h.clock,
                    stat=h.stat,
                    major=h.major,
                    minor=h.minor,
                    emit=print,
                )
            finally:
                sys.setprofile(previous)

        class Environ:
            get = staticmethod(h.getenv)

            def __getitem__(self, key):
                return h.getitem_env(key)

        with patch.object(
            test_recorded_calibration, "observe_f", test_recorded_calibration.observe_candidate
        ):
            yield SimpleNamespace(
                model=model,
                turn=turn,
                script=lambda argv: client.run_client(
                    arguments(client.ClientArguments, argv), turn=turn, open_file=h.open, emit=print
                ),
                bootstrap=bootstrap,
                conversation=lambda argv: run_conversation(
                    arguments(ConversationArguments, argv),
                    controller_factory=model.Controller,
                    turn=turn,
                    open_file=h.open,
                    environ=Environ(),
                    clock=h.clock,
                ),
                report=lambda argv: run_report(
                    argv[0],
                    machine_factory=model.make_machine,
                    open_file=h.open,
                    glob_paths=h.glob,
                    exists=h.exists,
                    basename=lambda p: p.rsplit("/", 1)[-1],
                    read_lines=lambda p: lines(p, h),
                    emit=print,
                ),
            )


@pytest.mark.parametrize("case", data()["cases"], ids=lambda c: c["name"])
def test_request_candidate_complete(case):
    assert observe(case, candidate_api) == case["expected"][runtime_key()]


def test_execution_ports_have_no_default():
    from inferssd.bootstrap.client import run_client, turn
    from inferssd.bootstrap.requests import post, run_bootstrap
    from inferssd.loop.conversation import run_conversation
    from inferssd.loop.report import run_report

    for function in (turn, post, run_client, run_bootstrap, run_conversation, run_report):
        ports = [
            p
            for p in inspect.signature(function).parameters.values()
            if p.kind is inspect.Parameter.KEYWORD_ONLY
        ]
        assert ports
        assert all(p.default is inspect.Parameter.empty for p in ports)


def test_offline_report_cli_matches_whole_f(tmp_path):
    import json
    import subprocess

    from test_loop_model import model_files

    case = next(
        c for c in data()["cases"] if c["name"] == "report-complete-static-routes-placement"
    )
    files = {"/case/run/" + key: json.dumps(value) for key, value in model_files().items()}
    files.update(case["files"])
    for path, text in files.items():
        target = tmp_path / path.removeprefix("/case/run/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    result = subprocess.run(
        [sys.executable, "-m", "inferssd", "report", str(tmp_path), "--host-threads", "8"],
        capture_output=True,
        text=True,
        check=True,
    )
    expected = case["expected"][runtime_key()]
    assert result.stdout == expected["stdout"]
    assert result.stderr == ""
    assert (tmp_path / "compare-online.json").read_text() == expected["files"][
        "/case/run/compare-online.json"
    ]


def test_request_modules_import_without_effects():
    import subprocess

    script = """
import builtins, subprocess, socket, time, urllib.request

def denied(*a, **kw):
    raise AssertionError('unexpected effect during import')

builtins.open = denied
subprocess.Popen = denied
socket.create_connection = denied
urllib.request.urlopen = denied
time.time = denied
time.time_ns = denied
import inferssd.bootstrap.client
import inferssd.bootstrap.requests
import inferssd.loop.conversation
import inferssd.loop.report
print('inert')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout == "inert\n"
    assert result.stderr == ""
