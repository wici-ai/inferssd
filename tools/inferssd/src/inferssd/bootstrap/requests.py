"""Complete bootstrap request protocol; all transport/host effects are required ports."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, cast
from urllib.request import Request

from inferssd.bootstrap.ports import Emit, FileDevice, History, OpenText, Post, Transport, Turn
from inferssd.telemetry.types import Record


@dataclass(frozen=True)
class BootstrapArguments:
    port: int
    script: str
    out: str
    max_tokens: int = 512
    tokenize: Optional[str] = None
    tokenize_out: Optional[str] = None
    drive: Optional[str] = None
    graph: bool = False


def drive_bytes(devices: set[tuple[int, int]], *, open_file: OpenText) -> int:
    """Bytes read so far by the block devices (major, minor), from /proc/diskstats (sectors of
    512 bytes).
    """
    total = 0
    for line in open_file("/proc/diskstats"):
        f = line.split()
        if (int(f[0]), int(f[1])) in devices:
            total += int(f[5]) * 512
    return total


def post(port: int, path: str, body: Record, *, transport: Transport) -> Record:
    req = Request(
        f"http://127.0.0.1:{port}{path}",
        json.dumps(body, ensure_ascii=False).encode(),
        {"Content-Type": "application/json"},
    )
    with transport(req, timeout=7200) as r:
        return cast(Record, json.loads(r.read()))


def run_bootstrap(
    a: BootstrapArguments,
    *,
    turn: Turn,
    post: Post,
    open_file: OpenText,
    clock: Callable[[], float],
    stat: Callable[[str], FileDevice],
    major: Callable[[int], int],
    minor: Callable[[int], int],
    emit: Emit,
) -> None:
    text = json.load(open_file(a.script, encoding="utf-8"))[0]["text"]
    msgs: History = [{"role": "user", "content": text}]

    def prompt_of(t: str) -> str:
        return cast(
            str,
            post(a.port, "/apply-template", {"messages": [{"role": "user", "content": t}]})[
                "prompt"
            ],
        )

    def graph_prefills() -> None:
        """--graph: the cut, then the whole text, each prefilled from scratch (one token of
        decode); the whole text alone when the trace covers it.
        """

        def tokens(t: str) -> int:
            return len(post(a.port, "/tokenize", {"content": prompt_of(t)})["tokens"])

        ends = [i + 1 for i in range(len(text) - 1) if text[i] in ".?!" and text[i + 1].isspace()]
        # the cut at ends[lo] fits the trace (none: -1), at ends[hi] it does not; none when the
        # whole text fits
        lo, hi = -1, 0 if tokens(text) <= a.max_tokens else len(ends)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if tokens(text[: ends[mid]]) <= a.max_tokens else (lo, mid)
        texts = ([text[: ends[lo]]] if lo >= 0 else []) + [text]
        with open_file(a.out, "w", encoding="utf-8") as out:
            for i, t in enumerate(texts):
                t0 = clock()
                r = post(
                    a.port,
                    "/completion",
                    {
                        "prompt": prompt_of(t),
                        "n_predict": 1,
                        "cache_prompt": False,
                        "temperature": 0,
                        "top_k": 1,
                    },
                )
                rec = {
                    "turn": i,
                    "kind": "graph_prefill",
                    "timings": r.get("timings", {}),
                    "wall_s": clock() - t0,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                emit(
                    f"graph prefill: prompt_n={rec['timings'].get('prompt_n')} "
                    f"prompt_ms={rec['timings'].get('prompt_ms')}",
                    flush=True,
                )

    if a.graph:
        graph_prefills()
        raise SystemExit(0)
    # the drives that hold the expert files: a file on a block device's filesystem has that device's
    # numbers (none for
    # an overlay or a network filesystem, which /proc/diskstats does not list)
    devices: set[tuple[int, int]] = set()
    if a.drive:
        for p in json.load(open_file(a.drive))["files"]:
            d = stat(p).st_dev
            devices.add((major(d), minor(d)))
        with open_file("/proc/diskstats") as f:
            devices &= {(int(x.split()[0]), int(x.split()[1])) for x in f}
    drive: dict[int, int] = {}

    def on_token(n: int) -> None:
        if devices and n in (1, a.max_tokens):
            drive[n] = drive_bytes(devices, open_file=open_file)

    with open_file(a.out, "w", encoding="utf-8") as out:
        rec = {"turn": 0, "kind": "chat", **turn(a.port, msgs, a.max_tokens, on_token)}
        if len(drive) == 2:
            rec["drive_read_bytes"] = drive[a.max_tokens] - drive[1]
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        t = rec["timings"]
        emit(
            f"chat: prompt_n={t.get('prompt_n')} prompt_ms={t.get('prompt_ms')} "
            f"predicted_n={t.get('predicted_n')}",
            flush=True,
        )
        prompt = post(a.port, "/apply-template", {"messages": msgs})["prompt"] + rec["reply"]
        t0 = clock()
        r = post(
            a.port,
            "/completion",
            {"prompt": prompt, "n_predict": 1, "cache_prompt": False, "temperature": 0, "top_k": 1},
        )
        rec2 = {
            "turn": 1,
            "kind": "warm_prefill",
            "timings": r.get("timings", {}),
            "wall_s": clock() - t0,
        }
        out.write(json.dumps(rec2, ensure_ascii=False) + "\n")
        t = rec2["timings"]
        emit(
            f"warm prefill: prompt_n={t.get('prompt_n')} prompt_ms={t.get('prompt_ms')}", flush=True
        )
    if a.tokenize:
        # a turn's new prompt tokens: the conversation rendered through it less the conversation
        # through the turn
        # before, the replies empty (a reply's tokens are decode, and the server keeps them)
        counts: list[int]
        hist: History
        counts, hist, total = [], [], 0
        for entry in json.load(open_file(a.tokenize, encoding="utf-8")):
            hist.append({"role": "user", "content": entry["text"]})
            p = post(a.port, "/apply-template", {"messages": hist})["prompt"]
            n = len(post(a.port, "/tokenize", {"content": p})["tokens"])
            counts.append(n - total)
            total = n
            hist.append({"role": "assistant", "content": ""})
        json.dump(counts, open_file(a.tokenize_out, "w"))
        emit(f"workload prompt tokens per turn: {counts}", flush=True)
