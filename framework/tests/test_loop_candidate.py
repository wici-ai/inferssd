"""Candidate comparison to the independently committed full reference transcripts."""

from __future__ import annotations

import contextlib
import json
from types import SimpleNamespace

import pytest
from test_loop_model import Effects, data, observe, runtime


@contextlib.contextmanager
def candidate_api(h):
    from inferssd.loop import model
    from inferssd.loop.controller import Controller
    from inferssd.loop.records import ObservationSession
    from inferssd.model.machine import Machine

    class Artifacts:
        def __init__(self, run):
            self.run = run

        def read_json(self, name):
            return json.load(h.open(f"{self.run}/{name}"))

        def exists(self, name):
            return h.exists(f"{self.run}/{name}")

    def machine(run):
        h.event("machine", run)
        m = Machine(Artifacts(run), host_threads=lambda: 8)
        h.machines.append(m)
        return m

    files = SimpleNamespace(
        open=h.open,
        gzip_open=h.gzip_open,
        exists=h.exists,
        stat=h.stat,
        open_binary=lambda p: h.open(p, "rb"),
    )
    session = ObservationSession(files=files)
    effects = SimpleNamespace(
        open=h.open,
        makedirs=h.makedirs,
        getenv=h.getenv,
        clock=h.clock,
        stdout=SimpleNamespace(write=h.output, flush=lambda: h.event("stdout_flush")),
    )

    def controller(*args, **kwargs):
        instance = Controller.__new__(Controller)
        h.controllers.append(instance)
        Controller.__init__(
            instance,
            *args,
            machine_factory=machine,
            engine_factory=h.engine,
            post=h.post,
            effects=effects,
            session=session,
            **kwargs,
        )
        return instance

    yield SimpleNamespace(
        online=SimpleNamespace(
            served=lambda *args: model.served(*args, session=session),
            fit_slope=model.fit_slope,
            feedback=model.feedback,
            expect=model.expect,
            decide=model.decide,
        ),
        make_machine=machine,
        Controller=controller,
        cache_read=session.read_arm_cached,
        arms=session.arms,
        live=session.live,
        session=session,
        traced={v.__code__: "machine." + k for k, v in vars(Machine).items() if callable(v)},
    )


@pytest.mark.parametrize("case", data()["cases"], ids=lambda c: c["name"])
def test_candidate_loop(case):
    assert observe(case, candidate_api) == case["expected"][runtime()]


def test_sessions_are_explicit_and_independent():
    """Isolation is the approved new ownership boundary, not old global semantics."""
    from inferssd.loop.records import ObservationSession

    with pytest.raises(TypeError):
        ObservationSession()
    h = Effects({"files": {"/case/a/server.log": ""}})
    with candidate_api(h) as one, candidate_api(h) as two:
        assert one.arms is not two.arms
        assert one.live is not two.live
        one.cache_read("/case/a", 0)
        assert list(one.live) == ["/case/a/server.log"]
        assert two.live == {}
        two.cache_read("/case/a", 0)
        assert one.live["/case/a/server.log"][0] is not two.live["/case/a/server.log"][0]


def test_controller_requires_all_effect_ports():
    import inspect

    from inferssd.loop.controller import Controller

    parameters = inspect.signature(Controller).parameters
    for name in ("machine_factory", "engine_factory", "post", "effects", "session"):
        assert parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default == inspect.Parameter.empty
    with pytest.raises(TypeError):
        Controller("run", "out", 1, (1, "all"), [])


def test_import_and_session_construction_have_no_effects():
    import subprocess
    import sys

    script = r"""
import builtins, os, pathlib, socket, subprocess, sys
# Preload standard library dependencies before trapping application effects.
import gzip, dataclasses, collections, statistics, typing, argparse, json

def denied(*args, **kwargs):
    raise AssertionError("unexpected implicit external effect")

builtins.open = denied
os.stat = denied
os.makedirs = denied
os.cpu_count = denied
socket.socket = denied
subprocess.Popen = denied
pathlib.Path.open = denied
from inferssd.loop import controller, model, records, ports
class Files:
    open = gzip_open = exists = stat = open_binary = denied
session = records.ObservationSession(files=Files())
assert session.arms == {} and session.live == {}
assert not hasattr(model, "_ARMS") and not hasattr(model, "_LIVE")
assert "controller" not in sys.modules and "online" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
