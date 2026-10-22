"""F's controller transitions with mandatory effect, engine and HTTP ports.

No native executor or live entry point is installed. Explicit close preserves F:
it stops the engine and does not close the event stream.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable, Sequence
from typing import Optional, cast

from ssdash.loop import model as online
from ssdash.loop.ports import Effects, Engine, EngineFactory, MachineFactory, Post, Segments
from ssdash.loop.records import ObservationSession
from ssdash.telemetry.types import Record


class Controller:
    """run: the bootstrap's directory (the machine model); out: this controller's directory;
    start: (tier, cpus). loop False keeps the start configuration (a static baseline).
    horizon: turns the re-plan looks ahead when the rest of the conversation is not known;
    decode: reply length assumed before any reply was served.
    """

    insts: list[str]
    asked: list[int]
    switch_s: list[float]
    engine: Optional[Engine]
    seen: Optional[tuple[int, list[Record], Segments]]

    def __init__(
        self,
        run: str,
        out: str,
        port: int,
        start: tuple[int, str],
        args: Iterable[str],
        loop: bool = True,
        window: tuple[int, int] = (129, 512),
        decode: int = 512,
        horizon: int = 3,
        *,
        machine_factory: MachineFactory,
        engine_factory: EngineFactory,
        post: Post,
        effects: Effects,
        session: ObservationSession,
    ) -> None:
        (self._engine_factory, self._post) = (engine_factory, post)
        (self._effects, self._session) = (effects, session)
        self.m = machine_factory(run)
        (self.out, self.port, self.args, self.loop) = (out, port, list(args), loop)
        (self.w0, self.w1) = window
        (self.decode, self.horizon) = (decode, horizon)
        self.keeps_history = self._effects.getenv("SWITCH_KEEPS_HISTORY", "1") == "1"
        self._effects.makedirs(out, exist_ok=True)
        self.events = self._effects.open(f"{out}/events.jsonl", "a")
        (self.insts, self.asked, self.n_turns, self.switch_s) = ([], [], 0, [])
        (self.engine, self.seen) = (None, None)
        self._start(*start)
        self.prepare()

    def note(self, rec: Record) -> None:
        rec["wall"] = self._effects.clock()
        self.events.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.events.flush()
        print(
            json.dumps({k: v for (k, v) in rec.items() if k != "decision"}),
            file=self._effects.stdout,
            flush=True,
        )

    def _start(self, tier: int, cpus: str) -> None:
        d = f"{self.out}/inst{len(self.insts)}"
        self.engine = self._engine_factory(
            d, self.port, tier, cpus, self.m.env(tier, cpus), self.args
        )
        self.insts.append(d)
        self.note(
            {
                "event": "start",
                "inst": len(self.insts) - 1,
                "tier": tier,
                "cpus": cpus,
                "env": self.engine.envs,
                "ok": self.engine.ok,
                "start_s": round(self.engine.start_s, 1),
            }
        )
        if not self.engine.ok:
            self.engine.stop()
            raise RuntimeError(
                f"llama-server did not come up in {tier},{cpus} (log: {d}/server.log.gz)"
            )

    @property
    def config(self) -> tuple[int, str]:
        return (cast(Engine, self.engine).tier, cast(Engine, self.engine).cpus)

    def before(self, n_new: int, remaining: Optional[Sequence[int]] = None) -> Optional[Record]:
        """A request with n_new new prompt tokens arrives. remaining: every turn's new prompt
        tokens from this one on when the conversation is known (validation), else the rest is
        expected from the turns so far. Returns the decision, None before the first turn has
        been served or without the loop.
        """
        self.ensure()
        if not self.loop or not self.n_turns:
            return None
        (turns, segs) = self.prepare()
        if remaining is None:
            (remaining, dec) = online.expect(self.asked, n_new, self.horizon, turns, self.decode)
        else:
            dec = self.decode
        cost = (
            statistics.mean(self.switch_s) if self.switch_s else cast(Engine, self.engine).start_s
        )
        d = online.decide(
            self.m,
            turns,
            segs,
            self.config,
            remaining,
            dec,
            self.w0,
            self.w1,
            cost,
            self.keeps_history,
        )
        self.note(
            {
                "event": "decision",
                "before_turn": self.n_turns,
                "switch": d["switch"],
                "to": d["to"],
                "feedback": d["feedback"],
                "switch_cost_s": round(cost, 1),
                "decision": d,
            }
        )
        if d["switch"]:
            t0 = self._effects.clock()
            if self.keeps_history:
                try:
                    self._post(
                        self.port,
                        "/completion",
                        {
                            "prompt": "OK",
                            "n_predict": 1,
                            "cache_prompt": True,
                            "temperature": 0,
                            "top_k": 1,
                        },
                    )
                except OSError as exc:
                    self.note({"event": "displace_failed", "error": repr(exc)})
            cast(Engine, self.engine).stop()
            self._start(*d["to"])
            self.switch_s.append(self._effects.clock() - t0)
            self.note({"event": "switched", "to": d["to"], "switch_s": round(self.switch_s[-1], 1)})
        return d

    def ensure(self) -> None:
        """Start the server again in its configuration if it has died."""
        if not cast(Engine, self.engine).alive():
            self.note({"event": "engine_died", "inst": len(self.insts) - 1})
            cast(Engine, self.engine).stop()
            self._start(*self.config)

    def prepare(self) -> tuple[list[Record], Segments]:
        """Read what the served turns showed (online.served; the slope outside the sink refitted)
        and price decode in every configuration with the routing so far. Once per served turn:
        serve.py calls it when a response is out, so a request's arrival only has the decision
        to make (the engine logs a token before it sends it).
        """
        if self.seen is None or self.seen[0] != self.n_turns:
            (turns, segs, pts) = online.served(
                self.m, self.insts, self.w0, self.w1, session=self._session
            )
            online.fit_slope(self.m, pts)
            for tier, p in self.m.configs:
                self.m.sink(segs or self.m.routing, tier, p)
            self.seen = (self.n_turns, turns, segs)
        return (self.seen[1], self.seen[2])

    @property
    def context(self) -> int:
        """Context position after the last served turn (0 before any)."""
        return cast(int, self.seen[1][-1]["end_pos"]) if self.seen and self.seen[1] else 0

    def after(self, n_new: int, rec: Record) -> None:
        """The request has been served: rec has its 'timings' (llama-server's) and 'tok_s'
        (arrival time of every decode token, s after the request; empty when not streamed).
        """
        (tier, cpus) = self.config
        rec = {"turn": self.n_turns, "tier": tier, "cpus": cpus, **rec}
        with self._effects.open(f"{cast(Engine, self.engine).dir}/turns.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        t = rec.get("timings") or {}
        self.note(
            {
                "event": "turn",
                "turn": self.n_turns,
                "inst": len(self.insts) - 1,
                "tier": tier,
                "cpus": cpus,
                "prompt_n": t.get("prompt_n"),
                "cache_n": t.get("cache_n"),
                "prefill_s": round(t.get("prompt_ms", 0) / 1000.0, 2),
                "predicted_n": t.get("predicted_n"),
                "decode_tps": online.window_rate(
                    cast(Sequence[Optional[float]], rec.get("tok_s")), self.w0, self.w1
                ),
                "wall_s": round(rec.get("wall_s", 0), 2),
            }
        )
        self.asked.append(n_new)
        self.n_turns += 1

    def close(self) -> None:
        if self.engine:
            self.engine.stop()
