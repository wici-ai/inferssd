from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from inferssd.model.routing_replay import Replay
from inferssd.plan import placement, settings

CANDIDATE = SimpleNamespace(settings=settings, placement=placement, Replay=Replay)

FIXTURES = Path(__file__).parent / "fixtures"
ORACLES = FIXTURES / "oracle"
FUNCTIONS = (
    "configuration",
    "prefill_route",
    "cost_spec",
    "prefill_carried",
    "prefill_start",
    "depth",
    "read_gain",
    "read_spill",
    "rank_costs",
    "prefetch_depth",
    "context",
    "derive",
    "env",
    "render",
)
PLACEMENTS = ("groups_of", "pin_env", "split_env", "env")


def load_oracle(name):
    path = ORACLES / f"{name}_reference.py"
    spec = importlib.util.spec_from_file_location(f"settings_{name}_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Original modules retain their own imports; temporary aliases are restored before
# any test returns, and no production path is imported as the oracle.
REFERENCE_PREFILL, REFERENCE_SIM = load_oracle("prefill"), load_oracle("sim")
saved_path = sys.path[:]
try:
    REFERENCE_PLACEMENT = load_oracle("placement")
    with patch.dict(
        sys.modules, placement=REFERENCE_PLACEMENT, prefill=REFERENCE_PREFILL, sim=REFERENCE_SIM
    ):
        REFERENCE_SETTINGS = load_oracle("settings")
        with patch.dict(sys.modules, settings=REFERENCE_SETTINGS):
            REFERENCE_MACHINE = load_oracle("machine")
finally:
    sys.path[:] = saved_path
REFERENCE = SimpleNamespace(
    settings=REFERENCE_SETTINGS, placement=REFERENCE_PLACEMENT, Replay=REFERENCE_SIM.Replay
)


def exact(value):
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    if isinstance(value, dict):
        return {"items": [[exact(k), exact(v)] for k, v in value.items()]}
    if isinstance(value, tuple):
        result = {"tuple": [exact(v) for v in value]}
        if hasattr(value, "_fields"):
            result["fields"] = list(value._fields)
        return result
    if isinstance(value, list):
        return {"list": [exact(v) for v in value]}
    if isinstance(value, set):
        return {"set_iteration": [exact(v) for v in value]}
    return value


def digest(value):
    return hashlib.sha256(json.dumps(exact(value), ensure_ascii=False).encode()).hexdigest()


class ObservedDict(dict):
    def __init__(self, path, events):
        super().__init__()
        self.path, self.events = path, events

    def __getitem__(self, key):
        self.events.append(["getitem", self.path, key])
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.events.append(["get", self.path, key, exact(default)])
        return super().get(key, default)

    def __setitem__(self, key, value):
        self.events.append(["setitem", self.path, key, exact(value)])
        return super().__setitem__(key, value)


def watched(value, path, events, memo):
    if isinstance(value, (dict, list)) and id(value) in memo:
        return memo[id(value)]
    if isinstance(value, dict):
        result = ObservedDict(path, events)
        memo[id(value)] = result
        for k, v in value.items():
            dict.__setitem__(result, k, watched(v, f"{path}.{k}", events, memo))
        return result
    if isinstance(value, list):
        result = []
        memo[id(value)] = result
        result.extend(watched(v, f"{path}[{i}]", events, memo) for i, v in enumerate(value))
        return result
    return value


def replay_state(replay):
    return {
        "rows": replay.rows,
        "seg_lens": replay.done,
        "seg_rows": replay.seg_rows,
        "step": replay.tier.step,
        "hits": replay.v,
        "total": replay.n_exp,
        "resident": replay.tier.res,
        "heap": replay.tier.heap,
        "scores": replay.tier.ls,
        "admissions": replay.tier.admissions,
    }


def observe(api, case):
    events, host_events, replay_events = [], [], []
    raw = copy.deepcopy(case["arguments"])
    args = watched(raw, "args", events, {})
    before = digest(args)
    cache = {}
    machine = None
    if "machine" in args:
        machine = SimpleNamespace(**args.pop("machine"))
        machine._replays = cache
        calls = 0

        def replay(key, segs, tier):
            nonlocal calls
            calls += 1
            replay_events.append(["call", exact(key), tier, digest(segs)])
            if calls == case.get("replay_raise_on"):
                raise RuntimeError("configured replay failure")
            # Reference Machine.replay body, with its Replay dependency supplied by each arm.
            r = cache.get(key)
            try:
                if r is None or r.tier.cap != tier:
                    raise ValueError
                r.advance(segs)
            except ValueError:
                r = cache[key] = api.Replay(tier, machine.mc, machine.ins["skip"]).advance(segs)
            replay_events.append(["return", digest(replay_state(r))])
            return r

        machine.replay = replay
        dict.__setitem__(args, "m", machine)
    if case.get("pending_iterator"):
        dict.__setitem__(args, "pending", iter(args["pending"]))
    if case.get("spill_tuple"):
        dict.__setitem__(args, "spill", tuple(args["spill"]))
    if case["function"] in ("env", "render") and not case.get("placement"):
        dict.__setitem__(args, "settings", [api.settings.Setting(*s) for s in args["settings"]])
        if case.get("settings_iterator"):
            dict.__setitem__(args, "settings", iter(args["settings"]))
    if case.get("head_iterator"):
        dict.__setitem__(args, "head", iter(args["head"]))
    host = case.get("host", {})

    def open_host(path):
        host_events.append(["open", path])
        value = host.get("files", {}).get(path)
        if value is None:
            raise OSError("configured host file absent")
        return io.StringIO(value)

    def cpu_count():
        host_events.append(["cpu_count"])
        if host.get("raise"):
            raise RuntimeError("configured cpu-count failure")
        return host.get("cpus", 8)

    def host_fallback():
        # The original host branch, entered only at default_threads' fallback.
        try:
            cpus = set()
            for path in ("/sys/devices/cpu_core/cpus", "/sys/devices/cpu_atom/cpus"):
                cpus |= set(REFERENCE_PLACEMENT.parse_list(open_host(path).read()))
            if cpus:
                return len(cpus)
        except OSError:
            pass
        return cpu_count()

    fn_name = case["function"]
    names = {getattr(api.settings, n).__code__: n for n in FUNCTIONS}
    names.update({getattr(api.placement, n).__code__: "placement." + n for n in PLACEMENTS})
    callgraph = []

    def profile(frame, event, arg):
        if event == "call" and frame.f_code in names:
            callgraph.append(names[frame.f_code])

    previous = sys.getprofile()
    sys.setprofile(profile)
    kwargs = dict(args)
    saved_os = REFERENCE_PLACEMENT.os
    outcome = None
    try:
        with patch.object(REFERENCE_PLACEMENT, "open", open_host, create=True):
            REFERENCE_PLACEMENT.os = SimpleNamespace(cpu_count=cpu_count)
            if api is not REFERENCE:
                if fn_name == "configuration" or fn_name == "generate":
                    kwargs["host_threads"] = host_fallback
                elif case.get("placement") and fn_name in ("env", "split_env"):
                    kwargs["host_threads"] = host_fallback
            if fn_name == "generate":
                host_kwargs = {"host_threads": host_fallback} if api is not REFERENCE else {}
                out = api.settings.configuration(
                    machine, kwargs["tier"], kwargs["p"], **host_kwargs
                )
                if not case.get("legacy_no_derived"):
                    out += api.settings.derive(
                        machine,
                        kwargs["p"],
                        kwargs.get("pending", ()),
                        kwargs.get("workload"),
                        kwargs.get("ledgers", ()),
                    )
                outcome = {
                    "settings": out,
                    "env": api.settings.env(out),
                    "render": api.settings.render(out, kwargs.get("head", ())),
                    "replacement": out[0]._replace(pending=True) if out else None,
                }
            else:
                mod = api.placement if case.get("placement") else api.settings
                outcome = {"return": getattr(mod, fn_name)(**kwargs)}
                if case.get("repeat"):
                    outcome["repeat"] = getattr(mod, fn_name)(**kwargs)
    except (
        KeyError,
        TypeError,
        ValueError,
        ZeroDivisionError,
        IndexError,
        RuntimeError,
        OverflowError,
        AttributeError,
    ) as error:
        outcome = {"exception": [type(error).__name__, str(error)]}
    finally:
        REFERENCE_PLACEMENT.os = saved_os
        sys.setprofile(previous)
    if machine is not None:
        args.pop("m")
        dict.__setitem__(
            args,
            "machine",
            {k: v for k, v in vars(machine).items() if k not in ("replay", "_replays")},
        )
    # Iterator state is observed by explicit consumption, not converted before the call.
    remaining = {}
    for key in ("pending", "settings", "head"):
        if key in args and hasattr(args[key], "__next__"):
            remaining[key] = list(args[key])
            dict.__setitem__(args, key, remaining[key])
    return {
        "outcome": exact(outcome),
        "input_before": before,
        "input_after": digest(args),
        "mapping_count": len(events),
        "mapping_sha256": digest(events),
        "mapping_edges": events[:5] + events[-5:],
        "host": host_events,
        "replay": replay_events,
        "cache": exact([(k, replay_state(v)) for k, v in cache.items()]),
        "remaining": exact(remaining),
        "calls": callgraph,
    }


GOLDENS = json.loads((FIXTURES / "settings-generation-goldens.json").read_text())
RUNTIME = f"{sys.version_info.major}.{sys.version_info.minor}"


@pytest.mark.parametrize("case", GOLDENS["cases"], ids=lambda case: case["name"])
def test_frozen_settings_generation(case):
    if RUNTIME not in ("3.9", "3.12"):
        pytest.skip("recorded reference transcripts cover Python 3.9 and 3.12 only")
    assert observe(REFERENCE, case) == case["expected"][RUNTIME]


def test_frozen_sources_and_namedtuple_contract():
    manifest = json.loads((ORACLES / "settings-manifest.json").read_text())
    for name, source in manifest["files"].items():
        assert (
            hashlib.sha256((ORACLES / f"{name}_reference.py").read_bytes()).hexdigest()
            == source["sha256"]
        )
    assert REFERENCE.settings.Setting._fields == ("name", "value", "source", "input", "pending")
    assert REFERENCE.settings.Setting._field_defaults == {"pending": False}


@pytest.mark.parametrize("case", GOLDENS["cases"], ids=lambda case: case["name"])
def test_complete_settings_matches_reference(case):
    assert observe(CANDIDATE, case) == observe(REFERENCE, case)


def test_setting_tuple_semantics_are_preserved():
    expected = REFERENCE_SETTINGS.Setting("X", 0, "derived", "receipt")
    actual = settings.Setting("X", 0, "derived", "receipt")
    assert actual == expected
    assert repr(actual) == repr(expected)
    assert actual._fields == expected._fields
    assert actual._field_defaults == expected._field_defaults
    assert actual._asdict() == expected._asdict()
    assert actual._replace(pending=True) == expected._replace(pending=True)
    with pytest.raises(AttributeError):
        actual.pending = True


@pytest.mark.parametrize(
    "case",
    [
        case
        for case in GOLDENS["cases"]
        if case["function"] == "rank_costs"
        and "error" not in case["name"]
        and case["name"] != "rank-replay-failure"
    ],
    ids=lambda case: case["name"],
)
def test_observed_cache_matches_original_machine_method(case):
    # Verify the observer's thin replay wrapper against the unchanged reference method,
    # rather than using the same fixture implementation as its own reference.
    from types import MethodType

    args = copy.deepcopy(case["arguments"])
    m = SimpleNamespace(**args.pop("machine"), _replays={})
    m.replay = MethodType(REFERENCE_MACHINE.Machine.replay, m)
    try:
        REFERENCE_SETTINGS.rank_costs(m, **args)
        if case.get("repeat"):
            REFERENCE_SETTINGS.rank_costs(m, **args)
    except ZeroDivisionError:
        assert case["name"] == "rank-zero-layers"
    expected = exact([(k, replay_state(v)) for k, v in m._replays.items()])
    assert observe(REFERENCE, case)["cache"] == expected
