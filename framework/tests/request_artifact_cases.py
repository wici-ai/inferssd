"""Synthetic inputs only. Expected observations come from whole frozen reference scripts."""

from __future__ import annotations

import copy
import json
import math

from test_loop_model import model_files
from test_request_artifacts import chat_frames, frame


def report_files():
    conv = {"event": "conversation", "wall": 10.0}
    turns = [
        {
            "event": "turn",
            "turn": i,
            "tier": 1,
            "cpus": "all",
            "prompt_n": n,
            "cache_n": 0,
            "prefill_s": 0.25 + i,
            "decode_tps": 20.0,
            "wall_s": 3.0,
            "wall": 12.0 + i * 4,
        }
        for i, n in enumerate([32, 17])
    ]
    plan = {
        "workload": {"decode": 20},
        "table": [
            {
                "tier": 1,
                "cpus": "all",
                "total_s": 4.0,
                "turns": [{"prefill_s": 0.5, "decode_tps": 20.0}],
            }
        ],
        "plan": {"tier": 1, "cpus": "all", "env": ""},
    }
    logs = (
        "\n".join(
            [
                "[kimi_cpu_prefill_graph] event=activate requested_tokens=32 configured_max=128",
                "[moe_prefill_copy] bytes=4096 read_us=50 passes=1 tokens=32",
                "[kimi_cpu_prefill_graph] event=activate requested_tokens=17 configured_max=128",
                "[kimi_decode_place] reason=destroy gpu_per_token=1 refused=2 "
                "error_p50_pct=15 error_p90_pct=18 cpu_moved=1.2",
                "[kimi_host_register] reason=destroy registered=2 failures=0",
                "[kimi_vram_host] copy_us_per_mb=250 resident_us_per_expert=1 host_us_per_expert=2",
                "[kimi_vram_admission] min_count=3 upload_us=45",
            ]
        )
        + "\n"
    )
    device = model_files()["probe/device.json"]
    device["gpu"]["h2d_pinned_gbs"] = 8
    return {
        "/case/run/plan.json": json.dumps(plan),
        "/case/run/conv-static/events.jsonl": "".join(
            json.dumps(x) + "\n" for x in [conv, *turns, {"event": "done", "wall": 18.0}]
        ),
        "/case/run/conv-static/inst0/server.log": logs,
        "/case/run/conv-static/drive.json": 'warning\n{"jobs":[{"read":{"bw_bytes":2147483648}}]}',
        "/case/run/probe/device.json": json.dumps(device),
    }


def cases():
    out = []

    def add(name, kind, **kw):
        out.append({"name": name, "kind": kind, **kw})

    add("turn-regular-unicode", "turn")
    events = [
        ": keepalive\n",
        frame({"choices": [{"delta": {"content": "前"}}]}),
        frame({"timings": {"predicted_n": 3}, "choices": [{"delta": {"content": "三"}}]}),
        frame({"timings": {"predicted_n": True}, "choices": [{"delta": None}]}),
        frame({"timings": {"predicted_n": 3}, "choices": [{"delta": {"content": "后"}}]}),
        frame({"timings": {"predicted_n": "4"}, "choices": []}),
        frame({"timings": {}, "choices": None}),
        "data: [DONE]\n",
        "data: not JSON\n",
    ]
    add("turn-sparse-duplicate-bool-content", "turn", responses=[{"lines": events}])
    add(
        "turn-missing-first",
        "turn",
        responses=[{"lines": [frame({"timings": {"predicted_n": 4}})]}],
    )
    add(
        "turn-no-timing",
        "turn",
        responses=[{"lines": [frame({"choices": [{"delta": {"content": "x"}}]})]}],
    )
    add("turn-empty", "turn", responses=[{"lines": []}])
    add("turn-no-callback", "turn", callback=False)
    add("turn-callback-mutates-history", "turn", callback_mutate=True, max_tokens=2)
    add("turn-bad-json", "turn", responses=[{"lines": [frame({}), "data: bad\n"]}])
    add(
        "turn-bad-content",
        "turn",
        responses=[{"lines": [frame({"choices": [{"delta": {"content": 7}}]})]}],
    )
    for op, key in (
        ("urlopen", "*"),
        ("http_enter", "0"),
        ("http_next", "0"),
        ("http_exit", "0"),
        ("callback", "2"),
        ("time_ns", ""),
    ):
        add("turn-fault-" + op, "turn", fault=[op, key, 1])

    add("script-history", "script")
    add("script-fresh", "script", argv=["--fresh"])
    add("script-empty", "script", files={"/case/script.json": "[]"})
    add(
        "script-second-error",
        "script",
        responses=[{"lines": chat_frames()}, {"lines": ["data: bad\n"]}],
    )
    add("script-entry-key-error-outside-catch", "script", files={"/case/script.json": "[{}]"})
    add("script-input-invalid-after-output-open", "script", files={"/case/script.json": "{"})
    for op, key in (
        ("open", "/case/script.json"),
        ("write", "/case/client.jsonl"),
        ("flush", "/case/client.jsonl"),
        ("close", "/case/client.jsonl"),
        ("stdout", "*"),
        ("stdout_flush", ""),
    ):
        add("script-fault-" + op, "script", fault=[op, key, 1])

    add("bootstrap-cold-warm", "bootstrap")
    add(
        "bootstrap-tokenize",
        "bootstrap",
        argv=["--tokenize", "/case/script.json", "--tokenize-out", "/case/counts.json"],
    )
    add(
        "bootstrap-tokenize-negative-counts",
        "bootstrap",
        argv=["--tokenize", "/case/script.json", "--tokenize-out", "/case/counts.json"],
        responses=[
            None,
            None,
            None,
            None,
            {"json": {"tokens": [1, 2, 3]}},
            None,
            {"json": {"tokens": []}},
        ],
    )
    for name, text, maximum in (
        ("whole", "a.", 20),
        ("cut", "One. Two? Three! Four.", 12),
        ("no-end", "longtext", 2),
        ("empty", "", 0),
        ("negative", "a. b. c.", -1),
    ):
        add(
            "bootstrap-graph-" + name,
            "bootstrap",
            argv=["--graph", "--max-tokens", str(maximum)],
            files={"/case/script.json": json.dumps([{"text": text}])},
        )
    drive = {"/case/experts.json": '{"files":["/case/a.gguf","/case/a.gguf","/case/b.gguf"]}'}
    for name, devices, maximum in (
        ("duplicate", {"/case/a.gguf": 2048, "/case/b.gguf": 2049}, 20),
        ("missing-device", {"/case/a.gguf": 999, "/case/b.gguf": 999}, 20),
        ("same-endpoint", {"/case/a.gguf": 2048, "/case/b.gguf": 2048}, 1),
    ):
        add(
            "bootstrap-drive-" + name,
            "bootstrap",
            files=drive,
            devices=devices,
            argv=["--drive", "/case/experts.json", "--max-tokens", str(maximum)],
            diskstats=[
                "8 0 toy 0 0 100 0\n8 1 toy1 0 0 200 0\n",
                "8 0 toy 0 0 105 0\n8 1 toy1 0 0 210 0\n",
                "8 0 toy 0 0 125 0\n8 1 toy1 0 0 240 0\n",
            ],
        )
    add("bootstrap-warm-http-error-after-record", "bootstrap", fault=["http_read", "2", 1])
    add(
        "bootstrap-tokenize-final-write-fault",
        "bootstrap",
        argv=["--tokenize", "/case/script.json", "--tokenize-out", "/case/counts.json"],
        fault=["write", "/case/counts.json", 2],
    )
    add("bootstrap-graph-ignores-bad-drive", "bootstrap", argv=["--graph", "--drive", "/missing"])
    add(
        "producer-fit-plan-settings-controller-report",
        "bootstrap",
        pipeline=True,
        argv=["--max-tokens", "68"],
        prompt_n=64,
    )

    add("conversation-static", "conversation")
    add("conversation-loop-known", "conversation", mode="loop")
    add("conversation-loop-learned", "conversation", mode="loop", argv=["--workload", "learned"])
    add(
        "conversation-empty",
        "conversation",
        files={"/case/script.json": "[]", "/case/tokens.json": "[]"},
    )
    add("conversation-no-token-for-second", "conversation", files={"/case/tokens.json": "[32]"})
    add("conversation-oserror-closes", "conversation", fault=["http_next", "0", 1])
    add("conversation-non-oserror-no-close", "conversation", responses=[{"lines": ["data: bad\n"]}])
    add(
        "conversation-close-fault-no-done",
        "conversation",
        fault=["stop", "/case/run/conv-static/inst0", 1],
    )
    add(
        "conversation-after-fault-no-close",
        "conversation",
        fault=["write", "/case/run/conv-static/inst0/turns.jsonl", 1],
    )
    add(
        "conversation-switch-full-chain",
        "conversation",
        mode="loop",
        report=True,
        start_s=0.001,
        argv=["--start", "1,split", "--decode", "5000"],
        fixture_tokens=20,
        responses=[{"lines": chat_frames()} for _ in range(4)],
        files={
            **{k: v for k, v in report_files().items() if "/conv-" not in k},
            "/case/script.json": json.dumps([{"text": "One."}] * 4),
            "/case/tokens.json": "[32,32,32,32]",
        },
    )
    add(
        "conversation-server-args-order", "conversation", env={"SERVER_ARGS": " --one x  --two  y "}
    )

    def report(name, change=None, **kw):
        files = report_files()
        if change:
            change(files)
        add("report-" + name, "report", files=files, **kw)

    report("complete-static-routes-placement")
    report(
        "gzip-preferred",
        lambda f: f.update(
            {
                "/case/run/conv-static/inst0/server.log.gz": f[
                    "/case/run/conv-static/inst0/server.log"
                ]
            }
        ),
    )
    report(
        "drive-fallback",
        lambda f: f.update(
            {
                "/case/run/conv-static/drive.json": "invalid",
                "/case/run/conv-static/fio.json": '{"jobs":[{"read":{"bw_bytes":1073741824}}]}',
            }
        ),
    )
    report(
        "no-done",
        lambda f: f.update(
            {
                "/case/run/conv-static/events.jsonl": f[
                    "/case/run/conv-static/events.jsonl"
                ].rsplit("\n", 2)[0]
                + "\n"
            }
        ),
    )
    report("no-turns", lambda f: f.update({"/case/run/conv-static/events.jsonl": ""}))
    report("missing-events", globs={"/case/run/conv-*": ["/case/run/conv-empty"]})
    report("model-invalid-json-caught", lambda f: f.update({"/case/run/insitu.json": "{"}))
    report("model-null-uncaught", lambda f: f.update({"/case/run/insitu.json": "null"}))
    report("device-invalid-caught", lambda f: f.update({"/case/run/probe/device.json": "{"}))
    report("device-null-uncaught", lambda f: f.update({"/case/run/probe/device.json": "null"}))
    for name, changes in (
        ("zero-decode-late-divzero", {"decode_tps": 0.0}),
        ("missing-conversation", None),
        ("missing-turn-field", "cache_n"),
    ):

        def change(f, changes=changes):
            path = "/case/run/conv-static/events.jsonl"
            ev = [json.loads(x) for x in f[path].splitlines()]
            if changes is None:
                ev = ev[1:]
            else:
                for x in ev:
                    if x["event"] == "turn":
                        if isinstance(changes, str):
                            x.pop(changes)
                        else:
                            x.update(changes)
            f[path] = "".join(json.dumps(x) + "\n" for x in ev)

        report(name, change)
    report(
        "placement-over-boundaries",
        lambda f: f.update(
            {
                "/case/run/conv-static/inst0/server.log": f[
                    "/case/run/conv-static/inst0/server.log"
                ]
                .replace("error_p50_pct=15", "error_p50_pct=15.00001")
                .replace("failures=0", "failures=2")
                .replace("copy_us_per_mb=250", "copy_us_per_mb=250.001")
            }
        ),
    )
    report(
        "placement-no-measured-copy",
        lambda f: f.update(
            {
                "/case/run/conv-static/inst0/server.log": f[
                    "/case/run/conv-static/inst0/server.log"
                ].replace("copy_us_per_mb=250", "copy_us_per_mb=0")
            }
        ),
    )
    report("no-route-log", lambda f: f.pop("/case/run/conv-static/inst0/server.log"))
    report(
        "gzip-read-error",
        lambda f: f.update({"/case/run/conv-static/inst0/server.log.gz": ""}),
        fault=["gzip_open", "/case/run/conv-static/inst0/server.log.gz", 1],
    )
    for label, actual in (
        ("nominal-15", 0.04 / 1.15),
        ("just-over-15", math.nextafter(0.04 / 1.15, 0.0)),
    ):

        def change(f, actual=actual):
            path = "/case/run/conv-static/events.jsonl"
            ev = [json.loads(x) for x in f[path].splitlines()]
            ev[1]["prefill_s"] = actual
            f[path] = "".join(json.dumps(x) + "\n" for x in ev)

        report("route-" + label, change)

    def missing_pick(f):
        path = "/case/run/conv-static/events.jsonl"
        f[path] += (
            json.dumps(
                {
                    "event": "decision",
                    "before_turn": 2,
                    "decision": {"candidates": [], "to": [1, "all"]},
                }
            )
            + "\n"
        )

    report("decision-missing-pick", missing_pick)
    report("final-partial-write", fault=["write", "/case/run/compare-online.json", 10])
    report("final-implicit-close-ignored", fault=["close", "/case/run/compare-online.json", 1])
    report("stdout-fault-before-final-file", fault=["stdout", "*", 3])
    return copy.deepcopy(out)
