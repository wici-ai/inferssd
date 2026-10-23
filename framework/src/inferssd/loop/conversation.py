"""Validation conversation orchestration with no native or HTTP defaults.

The original OSError path closes, but this extraction intentionally does not
add finally-cleanup for other exceptions. Native ownership is a separate port.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Optional, Protocol

from inferssd.bootstrap.ports import History, OpenText, Turn
from inferssd.telemetry.types import Record


class ConversationController(Protocol):
    def note(self, event: Record, /) -> None: ...
    def before(self, prompt_n: int, remaining: Optional[Sequence[int]] = None, /) -> object: ...
    def after(self, prompt_n: int, result: Record, /) -> None: ...
    def close(self) -> None: ...


ControllerFactory = Callable[..., ConversationController]


@dataclass(frozen=True)
class ConversationArguments:
    boot: str
    script: str
    tokens: str
    mode: str
    start: str
    out: str
    decode: int = 512
    window: str = "129-512"
    workload: str = "known"
    horizon: int = 3


def run_conversation(
    a: ConversationArguments,
    *,
    controller_factory: ControllerFactory,
    turn: Turn,
    open_file: OpenText,
    environ: Mapping[str, str],
    clock: Callable[[], float],
) -> None:
    texts = [e["text"] for e in json.load(open_file(a.script, encoding="utf-8"))]
    ntok = json.load(open_file(a.tokens))
    tier, cpus = a.start.split(",", 1)
    port = int(environ["PORT"])
    ctl = controller_factory(
        a.boot,
        a.out,
        port,
        (int(tier), cpus),
        environ.get("SERVER_ARGS", "").split(),
        loop=a.mode == "loop",
        window=tuple(int(x) for x in a.window.split("-")),
        decode=a.decode,
        horizon=a.horizon,
    )
    ctl.note(
        {
            "event": "conversation",
            "mode": a.mode,
            "start": a.start,
            "turns": len(texts),
            "tokens": ntok,
            "workload": a.workload,
            "horizon": a.horizon if a.workload == "learned" else None,
        }
    )
    history: History
    history, t_conv = [], clock()
    for i, text in enumerate(texts):
        ctl.before(ntok[i], ntok[i:] if a.workload == "known" else None)
        history.append({"role": "user", "content": text})
        t0 = clock()
        try:
            rec = turn(port, history, a.decode)
        except OSError as exc:
            ctl.note({"event": "turn", "turn": i, "error": repr(exc)})
            break
        history.append({"role": "assistant", "content": rec["reply"]})
        ctl.after(ntok[i], {"wall_s": clock() - t0, **rec})
    ctl.close()
    ctl.note({"event": "done", "total_s": round(clock() - t_conv, 1)})
