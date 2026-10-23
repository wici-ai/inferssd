from __future__ import annotations

import ast
import contextlib
import copy
import gzip
import hashlib
import io
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
from test_offline_planner import F_PREFILL, F_SIM, exact, load

FIXTURES = Path(__file__).parent / "fixtures"
with patch.dict(sys.modules, prefill=F_PREFILL):
    F = load("logparse")


def consumer(name):
    """Load frozen definitions without executing insitu's unguarded CLI."""
    path = FIXTURES / "oracle" / f"{name}_f.py"
    tree = ast.parse(path.read_text())
    if name == "insitu":
        tree.body = tree.body[
            : next(
                i
                for i, n in enumerate(tree.body)
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "ap" for t in n.targets)
            )
        ]
    module = ModuleType(f"log_consumer_{name}_F")
    module.__file__ = str(path)
    saved = sys.path[:]
    try:
        with patch.dict(sys.modules, prefill=F_PREFILL, sim=F_SIM, logparse=F):
            exec(compile(tree, str(path), "exec"), module.__dict__)
    finally:
        sys.path[:] = saved
    return module


F_INSITU, F_ONLINE = consumer("insitu"), consumer("online")


def runtime_key():
    version = sys.version_info[:2]
    if version not in ((3, 9), (3, 12)):
        pytest.skip("frozen F transcripts cover Python 3.9 and 3.12 only")
    return f"{version[0]}.{version[1]}"


def digest(value):
    return hashlib.sha256(json.dumps(exact(value), ensure_ascii=False).encode()).hexdigest()


class Files:
    """Tiny synthetic filesystem; operations and first faults are observable."""

    def __init__(self, case):
        self.data = copy.deepcopy(case.get("files", {}))
        self.events = []
        self.fail = case.get("file_fault")

    def open(self, path, mode="r", *, errors=None):
        path = str(path)
        self.events.append(["open", path, mode, errors])
        if self.fail == ["open", path]:
            raise PermissionError("configured open failure")
        if path not in self.data:
            raise FileNotFoundError(f"missing synthetic file: {path}")
        owner = self

        class Stream(io.StringIO):
            def read(self, *args):
                owner.events.append(["read", path])
                if owner.fail == ["read", path]:
                    raise OSError("configured read failure")
                return super().read(*args)

            def __next__(self):
                owner.events.append(["next", path])
                if owner.fail == ["next", path]:
                    raise OSError("configured iteration failure")
                return super().__next__()

            def close(self):
                if not self.closed:
                    owner.events.append(["close", path])
                super().close()

        return Stream(self.data[path])

    def gzip_open(self, path, mode="r", *, errors=None):
        self.events.append(["gzip", str(path), mode, errors])
        if self.fail == ["gzip", str(path)]:
            raise gzip.BadGzipFile("configured invalid gzip")
        return self.open(path, mode, errors=errors)


@contextlib.contextmanager
def f_api(files):
    with (
        patch.object(F, "open", files.open, create=True),
        patch.object(F, "gzip", SimpleNamespace(open=files.gzip_open)),
    ):
        yield SimpleNamespace(
            ArmParser=F.ArmParser,
            result=lambda p, d, s: p.result(d, s),
            read_arm=F.read_arm,
            ledger=F.ledger,
            hits=F.hits,
            window_rate=F.window_rate,
            median=F.median,
        )


def aliases(p, result):
    keys = (
        "trace",
        "prof",
        "lprof",
        "admission",
        "threads",
        "host",
        "ledger",
        "mrc",
        "prefetch",
        "decode_place",
        "host_register",
        "cache_load",
    )
    names = {"admission": "adm"}
    same = {k: result[k] is getattr(p, names.get(k, k)) for k in keys}
    same["lio_outer"] = result["lio"] is p.lio
    same["lio_values"] = all(v is p.lio[k] for k, v in result["lio"].items())
    same["segment_values"] = all(
        v is p.trace[k] for g, s in zip(result["groups"], result["segments"]) for k, v in zip(g, s)
    )
    same["graph_values"] = all(any(v is g for g in p.pf_graphs) for v in result["pf_graphs"])
    same["reserves"] = result["memory"].get("reserves") is p.memory.get("reserves")
    return same


def observe(case, api_factory=f_api):
    files = Files(case)
    steps, snapshots = [], []
    parser = None
    with api_factory(files) as api:
        parser = api.ArmParser()
        for action in case["actions"]:
            kind = action[0]
            try:
                if kind == "feed":
                    parser.feed(action[1])
                    value = None
                elif kind == "result":
                    value = api.result(parser, "/case/run", action[1])
                    snapshots.append(value)
                    steps.append(["aliases", aliases(parser, value)])
                elif kind == "read_arm":
                    value = api.read_arm("/case/run", action[1], parser if action[2] else None)
                elif kind == "ledger":
                    value = api.ledger("/case/run")
                elif kind == "hits":
                    value = api.hits(snapshots[-1], action[1])
                elif kind == "window_rate":
                    value = api.window_rate(*action[1:])
                elif kind == "median":
                    value = api.median(action[1])
                elif kind in ("measure", "measure_graph"):
                    with (
                        patch.object(F_INSITU, "read_arm", api.read_arm),
                        patch.object(F_INSITU, "hits", api.hits),
                        patch.object(F_INSITU, "median", api.median),
                    ):
                        value = getattr(F_INSITU, kind)("/case/run", *action[1:])
                elif kind == "served":
                    calls = []

                    class Machine:
                        def replay(self, key, segments, tier):
                            calls.append(["replay", exact(key), exact(segments), tier])
                            return SimpleNamespace(
                                seg_rows=[(0, 1)] * len(segments), rows=[[1, 2, 3]]
                            )

                        def price(self, rows, cpus):
                            calls.append(["price", rows, cpus])
                            return 2.5, 1.25

                    with (
                        patch.object(F_ONLINE, "open", files.open, create=True),
                        patch.object(
                            F_ONLINE,
                            "os",
                            SimpleNamespace(path=SimpleNamespace(exists=lambda p: p in files.data)),
                        ),
                        patch.object(F_ONLINE, "_read_arm_cached", api.read_arm),
                        patch.object(F_ONLINE, "window_rate", api.window_rate),
                    ):
                        value = [F_ONLINE.served(Machine(), ["/case/run"], *action[1:]), calls]
                else:
                    raise AssertionError(kind)
                steps.append([kind, "return", exact(value), digest(vars(parser))])
            except Exception as error:
                steps.append([kind, type(error).__name__, str(error), digest(vars(parser))])
    return {
        "steps": steps,
        "state": exact(vars(parser)),
        "snapshots_after": exact(snapshots),
        "files": files.events,
    }


DATA = json.loads((FIXTURES / "log-records-goldens.json").read_text())


@pytest.mark.parametrize("case", DATA["cases"], ids=lambda c: c["name"])
def test_f_log_records(case):
    assert observe(case) == case["expected"][runtime_key()]


def test_f_log_source_identity():
    manifest = json.loads((FIXTURES / "oracle/logparse-manifest.json").read_text())
    for item in manifest["files"]:
        assert (
            hashlib.sha256((FIXTURES / "oracle" / item["file"]).read_bytes()).hexdigest()
            == item["sha256"]
        )


@contextlib.contextmanager
def candidate_api(files):
    from ssdash.telemetry import legacy, recorded

    yield SimpleNamespace(
        ArmParser=legacy.ArmParser,
        result=lambda p, d, s: p.result(d, s, files=files),
        read_arm=lambda d, s=128, p=None: recorded.read_arm(d, s, p, files=files),
        ledger=lambda d: recorded.ledger(d, files=files),
        hits=legacy.hits,
        window_rate=legacy.window_rate,
        median=legacy.median,
    )


@pytest.mark.parametrize("case", DATA["cases"], ids=lambda c: c["name"])
def test_candidate_log_records(case):
    assert observe(case, candidate_api) == case["expected"][runtime_key()]
