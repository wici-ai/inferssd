"""Legacy streamed-token observations, distinct from the formal benchmark timer."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional
from urllib.request import Request

from inferssd.bootstrap.ports import Emit, History, OpenText, Transport, Turn
from inferssd.telemetry.types import Record


@dataclass(frozen=True)
class ClientArguments:
    port: int
    script: str
    out: str
    max_tokens: int = 256
    fresh: bool = False


def turn(
    port: int,
    history: History,
    max_tokens: int,
    on_token: Optional[Callable[[int], None]] = None,
    *,
    transport: Transport,
    clock_ns: Callable[[], int],
) -> Record:
    """One streamed chat turn; on_token(n) is called as decode token n arrives."""
    body = {
        "model": "x",
        "messages": history,
        "max_tokens": max_tokens,
        "stream": True,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
        "timings_per_token": True,
    }
    req = Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(body, ensure_ascii=False).encode(),
        {"Content-Type": "application/json"},
    )
    tok: dict[int, int]
    parts: list[str]
    timings: Optional[Record]
    tok, parts, timings = {}, [], None
    t0 = clock_ns()
    with transport(req, timeout=7200) as r:
        for raw in r:
            if not raw.startswith(b"data: "):
                continue
            p = raw[6:].strip()
            if p == b"[DONE]":
                break
            ev = json.loads(p)
            now = clock_ns()
            if ev.get("timings"):
                timings = ev["timings"]
                n = timings.get("predicted_n")
                if isinstance(n, int) and n >= 1 and n not in tok:
                    tok[n] = now
                    if on_token:
                        on_token(n)
            for c in ev.get("choices") or []:
                d = (c.get("delta") or {}).get("content")
                if d:
                    parts.append(d)
    n = max(tok) if tok else 0
    return {
        "timings": timings or {},
        "reply": "".join(parts),
        "ttft_s": (tok[1] - t0) / 1e9 if 1 in tok else None,
        "tok_s": [round((tok[i] - t0) / 1e9, 6) if i in tok else None for i in range(1, n + 1)],
    }


def run_client(a: ClientArguments, *, turn: Turn, open_file: OpenText, emit: Emit) -> int:
    history: History = []
    with open_file(a.out, "w", encoding="utf-8") as out:
        for i, entry in enumerate(json.load(open_file(a.script, encoding="utf-8"))):
            if a.fresh:
                history = []
            history.append({"role": "user", "content": entry["text"]})
            try:
                rec = {"turn": i, **turn(a.port, history, a.max_tokens)}
                history.append({"role": "assistant", "content": rec["reply"]})
            except Exception as exc:  # noqa: BLE001 - recorded, the arm continues to cleanup
                rec = {"turn": i, "error": repr(exc)}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            t = rec.get("timings", {})
            emit(
                f"turn {i}: prompt_n={t.get('prompt_n')} prefill_tps={t.get('prompt_per_second')} "
                f"predicted_n={t.get('predicted_n')} {rec.get('error', '')}",
                flush=True,
            )
            if "error" in rec:
                return 1
    return 0
