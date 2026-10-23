"""Whole original F calibration program: synthetic records, state and I/O ordering."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_log_records import Files, f_api, runtime_key
from test_offline_planner import F_PREFILL, F_SIM, exact

FIXTURES = Path(__file__).parent / "fixtures"
ORACLE = FIXTURES / "oracle/insitu_f.py"
GOLDENS = FIXTURES / "recorded-calibration-goldens.json"


def synthetic_files(config):
    """Generate toy engine text, not expected calculations. Original F parses it."""
    n, layers = config.get("tokens", 68), config.get("layers", 2)
    lines = [
        "using device TOY0 synthetic - 8192 MiB free",
        "TOY0 model buffer size = 10.25 MiB",
        "TOY0 KV buffer size = 2.5 MiB",
        "n_ubatch = 128",
        "TOY0 compute buffer size = 4.5 MiB",
        "CPU compute buffer size = 1.5 MiB",
        "context checkpoints are taken after each prompt batch at multiples of 4 tokens",
        "[kimi_vram_tier] reason=allocated capacity_bytes=40960 slots=10 "
        "budget_bytes=20480 min_count=2",
        "[kimi_vram_elastic] event=decode_grow before_slots=10 after_slots=12 slots=12",
        "[kimi_host_ledger] reason=destroy unified=0 margin=1024 margin_updates=2",
        "[kimi_cpu_sink_threads] decode_participants=2 us_per_expert=4 (3 participants: 5)",
        "[kimi_cpu_sink_prefetch] enabled topk=1 readers=1",
        "[kimi_cpu_sink_prefetch] reason=destroy topk=1 layers=136 "
        "reader_bytes=200 window_us_p50=10 predicted_r=4,2 novel_r=3,1 "
        "novel_routed_r=2,1",
        "[kimi_cpu_sink_cache] auto_budget=8192 slot_bytes=4096",
        "[kimi_cpu_sink_cache] reason=load slots=2 committed_bytes=8192 us=200",
        '[kimi_cpu_sink_mrc] {"capacity":[0,2,4],"miss":[1,0.5,0.25]}',
        "[kimi_vram_host] resident_us_per_expert=1 host_us_per_expert=2 "
        "in_place=0 copy_us_per_mb=500",
        "[kimi_decode_place] reason=destroy gpu_per_token=1 refused=0 "
        "error_p50_pct=2 error_p90_pct=3",
        "[kimi_host_register] reason=destroy registered=2 failures=0 us_mean=4",
    ]
    if not config.get("no_lane"):
        lines += [
            "[kimi_vram_prefill_engine] format=toy engine=group in_place=0 "
            "route_bytes=8 token_bytes=4 output_bytes=2 fixed_bytes=128 "
            "bytes_per_token=16 reserved_rows=64"
        ]
    for size in config.get("prefills", [64, 128]):
        lines += [
            f"[kimi_cpu_prefill_graph] event=activate requested_tokens={size} configured_max=128",
            '[kimi_cpu_sink_profile] {"phase":"prefill","event":"layer","compute_wall_us":125000}',
            f"[moe_prefill_copy] bytes=4096 "
            f"read_us={config.get('graph_read_us', 12000)} passes=1 "
            f"tokens={size}",
        ]
        for layer in range(layers):
            lines += [
                f"[kimi_cpu_prefill_cost] calibrated=1 cpu_fixed_us=0.25 "
                f"gpu_fixed_us=0 link_gbps=2 measured_us=500 predicted_us=600 "
                f"tokens={size}",
                f"[kimi_cpu_prefill_layer] layer={layer} n_tokens={size} "
                f"unique_experts=3 routed_slots={size * 2} gpu_experts=1 "
                f"gpu_ssd_transient_experts=0 ssd_experts=2 ssd_bytes=2048",
            ]
    for token in range(n):
        for layer in range(layers):
            if config.get("partial") and token == 20 and layer == 1:
                continue
            cls = 0 if config.get("singular") else token % 4
            vh, ch = (
                ([True, True], [False, False])
                if cls == 0
                else (
                    ([True, False], [False, True])
                    if cls == 1
                    else (
                        ([False, False], [True, False])
                        if cls == 2
                        else ([False, False], [False, False])
                    )
                )
            )
            common = dict(phase="decode", event="layer", epoch=0, token=token, layer=layer)
            trace = dict(
                common,
                selected_experts=[(token + layer) % 4, (token + layer + 1) % 4],
                vram_hit=vh,
                cache_hit=ch,
            )
            profile = dict(
                common,
                total_us=1000 + token * 0.1 + layer * 0.25,
                wait_io_us=100,
                compute_wall_us=500,
                other_wall_us=200,
            )
            lines.append("[kimi_cpu_sink_trace] " + json.dumps(trace, separators=(",", ":")))
            if not config.get("no_profile"):
                lines.append(
                    "[kimi_cpu_sink_profile] " + json.dumps(profile, separators=(",", ":"))
                )
                lines.append(
                    f"[kimi_cpu_sink_layer] io_us={config.get('io_us', 10)} "
                    f"bytes={config.get('io_bytes', 4096)} "
                    f"cache_new_slot_bytes=4"
                )
    clients = [
        dict(
            kind="chat",
            drive_read_bytes=1000000,
            tok_s=[i * config.get("token_interval", 0.01) for i in range(n + 1)],
            timings=dict(prompt_n=64, prompt_ms=300),
        ),
        dict(kind="warm", timings=dict(prompt_n=128, prompt_ms=500)),
    ]
    if config.get("empty_times"):
        clients[0]["tok_s"] = []
    if config.get("null_times"):
        clients[0]["tok_s"][20] = None
        clients[0]["tok_s"][-1] = None
    for c, size in zip(clients, config.get("client_prefills", [])):
        c["timings"]["prompt_n"] = size
    if config.get("mismatch_graph"):
        clients.append(dict(kind="warm", timings=dict(prompt_n=64, prompt_ms=600)))
    if config.get("no_memory"):
        lines = [
            line for line in lines if not line.startswith(("using ", "TOY0 ", "CPU ", "n_ubatch"))
        ]
    return {
        "server.log": "\n".join(lines) + "\n",
        "client.jsonl": "".join(json.dumps(c) + "\n" for c in clients),
        "meta.txt": "load_s=1.25\n",
    }


def case_files(case):
    out = {}
    for run, config in case.get("runs", {"/case/a": {}}).items():
        out.update({run + "/" + name: value for name, value in synthetic_files(config).items()})
    out.update(case.get("files", {}))
    for name in case.get("remove", []):
        out.pop(name, None)
    return out


class CalibrationTranscript:
    def __init__(self, case):
        self.case = case
        self.logs = Files({"files": case_files(case), "file_fault": case.get("file_fault")})
        self.events = self.logs.events
        self.writes = {}
        self.output = io.StringIO()
        self.dump_calls = []
        self.arms, self.measured = [], []
        self.stdout_calls = 0

    def open(self, path, mode="r"):
        if mode != "w":
            return self.logs.open(path, mode)
        self.events.append(["open_output", path])
        if self.case.get("fail_open") == path:
            raise OSError("configured output open failure")
        self.writes[path] = ""
        owner = self

        class Writer(io.StringIO):
            count = 0

            def write(self, text):
                self.count += 1
                owner.events.append(
                    ["write_output", path, len(text), hashlib.sha256(text.encode()).hexdigest()]
                )
                if owner.case.get("fail_write") == [path, self.count]:
                    owner.writes[path] += text[:1]
                    raise OSError("configured output write failure")
                owner.writes[path] += text
                return len(text)

            def close(self):
                if not self.closed:
                    owner.events.append(["close_output", path])
                    super().close()
                    if owner.case.get("fail_close") == path:
                        raise OSError("configured output close failure")

            def __del__(self):
                # F gives json.dump an unbound temporary: close failure is ignored,
                # not converted to an explicit with-block exception.
                try:
                    self.close()
                except OSError as error:
                    owner.events.append(["ignored_close", path, str(error)])

        return Writer()

    def dump(self, obj, stream, **kwargs):
        self.dump_calls.append([exact(obj), exact(kwargs)])
        return json.dump(obj, stream, **kwargs)

    def stdout(self, text):
        self.stdout_calls += 1
        self.events.append(["stdout", text])
        if self.stdout_calls == self.case.get("fail_stdout"):
            self.output.write(text[:1])
            raise OSError("configured stdout failure")
        return self.output.write(text)

    def arm(self, read, run, skip):
        self.events.append(["read_arm", run, skip])
        value = read(run, skip)
        for key, replacement in self.case.get("arm_overrides", {}).get(run, {}).items():
            value[key] = copy.deepcopy(replacement)
        if self.case.get("nonserializable_mrc"):
            value["mrc"] = {1, 2}
        self.arms.append(value)
        return value

    def result(self, ns, outcome):
        keys = ("placements", "first", "routing", "n_exp", "prof", "ins")
        placements = ns.get("placements", {})
        aliases = {
            "first": bool(placements) and ns.get("first") is next(iter(placements.values())),
            "ins_placements": ns.get("ins", {}).get("placements") is placements,
            "profile_routing": ns.get("prof", {}).get("routing") is ns.get("routing"),
            "measured_placement": [any(m is p for p in placements.values()) for m in self.measured],
            "measured_routing_present": ["routing" in m for m in self.measured],
            "derived": [
                [name, key, p.get(key) is placements[p["derived_from"]].get(key)]
                for name, p in placements.items()
                if "derived_from" in p
                for key in (
                    "measured",
                    "layer_table",
                    "layer_table_raw",
                    "layer_beta",
                    "prefill",
                    "memory",
                )
            ],
            "arm_memory": [
                [i, m.get("memory") is a.get("memory")]
                for i, (m, a) in enumerate(zip(self.measured, self.arms))
            ],
        }
        return {
            "outcome": outcome,
            "events": self.events,
            "writes": self.writes,
            "stdout": self.output.getvalue(),
            "dump_calls": self.dump_calls,
            "state": exact({k: ns[k] for k in keys if k in ns}),
            "aliases": aliases,
        }


def argv(case):
    return case.get(
        "argv",
        [
            "--run",
            "all=/case/a",
            "--skip",
            "4",
            "--profile",
            "/case/model.json",
            "-o",
            "/case/insitu.json",
        ],
    )


def observe_f(case):
    tr, ns = CalibrationTranscript(case), {"__file__": str(ORACLE), "__name__": "__main__"}
    old_path, old_trace = sys.path[:], sys.gettrace()

    def trace(frame, event, arg):
        if (
            frame.f_code.co_filename == str(ORACLE)
            and frame.f_code.co_name == "measure"
            and event == "return"
            and arg is not None
        ):
            tr.measured.append(arg)
        return trace

    try:
        with (
            f_api(tr.logs) as api,
            patch.dict(
                sys.modules,
                prefill=F_PREFILL,
                sim=F_SIM,
                logparse=SimpleNamespace(
                    read_arm=lambda r, s: tr.arm(api.read_arm, r, s),
                    hits=api.hits,
                    median=api.median,
                ),
            ),
            patch.object(sys, "argv", ["insitu.py", *argv(case)]),
            patch.object(sys, "stdout", SimpleNamespace(write=tr.stdout)),
        ):
            ns["open"] = tr.open
            ns["json"] = None
            # Replace only this execution's imported json module, preserving json.dump's
            # implementation.
            with patch.dict(
                sys.modules, json=SimpleNamespace(load=json.load, dumps=json.dumps, dump=tr.dump)
            ):
                sys.settrace(trace)
                exec(compile(ORACLE.read_text(), str(ORACLE), "exec"), ns)
        outcome = ["return"]
    except Exception as error:
        outcome = [type(error).__name__, str(error)]
    finally:
        sys.settrace(old_trace)
        sys.path[:] = old_path
    result = tr.result(ns, outcome)
    if case.get("chain") and outcome == ["return"]:
        from test_offline_planner import observe_f as plan_observer

        result["chain"] = plan_observer(chain_case(case, result["writes"]))
    return result


def chain_case(case, writes):
    return {
        "files": {
            "insitu.json": writes["/case/insitu.json"],
            "model.json": writes["/case/model.json"],
            "probe/device.json": case["device"],
        },
        "argv": ["--prompts", "32,17", "--decode", "8", "--window", "1-8", "--known"],
    }


def data():
    return json.loads(GOLDENS.read_text())


@pytest.mark.parametrize("case", data()["cases"], ids=lambda c: c["name"])
def test_f_complete_calibration(case):
    assert observe_f(case) == case["expected"][runtime_key()]


def test_calibration_source_identity():
    assert (
        hashlib.sha256(ORACLE.read_bytes()).hexdigest()
        == "6b79f8a068b5228e56ae408327fb6787b13900ea5d6d92b58d60f98a13d021ee"
    )


def observe_candidate(case):
    import argparse

    from inferssd.model.observed_calibration import measure
    from inferssd.probe import calibrate
    from inferssd.telemetry.recorded import read_arm

    tr, ns = CalibrationTranscript(case), {}
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--skip", type=int, default=128)
    parser.add_argument("--profile", required=True)
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("--graph")
    parser.add_argument("--ubatch-probe")
    parsed = parser.parse_args(argv(case))
    a = calibrate.CalibrationArguments(**vars(parsed))
    old_trace = sys.gettrace()

    def trace(frame, event, arg):
        if event == "return" and frame.f_code == measure.__code__ and arg is not None:
            tr.measured.append(arg)
        if event == "return" and frame.f_code == calibrate.run_calibration.__code__:
            ns.update(frame.f_locals)
        return trace

    try:
        with (
            patch.object(
                calibrate, "json", SimpleNamespace(load=json.load, dumps=json.dumps, dump=tr.dump)
            ),
            patch.object(sys, "stdout", SimpleNamespace(write=tr.stdout)),
        ):
            sys.settrace(trace)
            calibrate.run_calibration(
                a,
                read_arm=lambda r, s: tr.arm(lambda r, s: read_arm(r, s, files=tr.logs), r, s),
                open_file=tr.open,
                emit=print,
            )
        outcome = ["return"]
    except Exception as error:
        outcome = [type(error).__name__, str(error)]
    finally:
        sys.settrace(old_trace)
    result = tr.result(ns, outcome)
    if case.get("chain") and outcome == ["return"]:
        from test_offline_planner import observe_candidate as plan_observer

        result["chain"] = plan_observer(chain_case(case, result["writes"]))
    return result


@pytest.mark.parametrize("case", data()["cases"], ids=lambda c: c["name"])
def test_candidate_complete_calibration(case):
    assert observe_candidate(case) == case["expected"][runtime_key()]
