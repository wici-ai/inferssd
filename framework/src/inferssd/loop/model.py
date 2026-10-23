"""F's served-turn and re-plan calculations, using explicit observation state."""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable, Sequence
from typing import Union, cast

from inferssd.loop.ports import Model, Point, Segments
from inferssd.loop.records import ObservationSession
from inferssd.model.machine_types import Feedback
from inferssd.model.prefill_feedback import ServedTurn
from inferssd.model.prefill_feedback import feedback as prefill_feedback
from inferssd.model.routing_replay import Routing
from inferssd.telemetry.legacy import window_rate as window_rate
from inferssd.telemetry.types import Record


def served(
    m: Model, insts: Iterable[str], w0: int, w1: int, *, session: ObservationSession
) -> tuple[list[Record], Segments, list[Point]]:
    """Turns served so far, in order, from the server instances' directories; plus every decode
    segment's routing (for the replay) and the outside-the-sink points (position, ms,
    placement) for the slope. A request's decode is the next trace segment that starts where
    its prompt ends (cache_n + prompt_n); its prefill the next prefill graph of its prompt's
    size.
    """
    (turns, segs, pts) = ([], [], [])
    for d in insts:
        recs = (
            [json.loads(x) for x in session.files.open(f"{d}/turns.jsonl")]
            if session.files.exists(f"{d}/turns.jsonl")
            else []
        )
        if not recs:
            continue
        a = session.read_arm_cached(d, w0 - 1)
        (groups, graphs, gk, gi, mine) = (a["groups"], a["pf_graphs"], 0, 0, [])
        for i, r in enumerate(recs):
            t = r["timings"]
            n = t.get("prompt_n") or 0
            start = (t.get("cache_n") or 0) + n
            j = next((j for j in range(gi, len(graphs)) if graphs[j]["n_tokens"] == n), None)
            pg = graphs[j] if j is not None else None
            gi = j + 1 if j is not None else gi
            q = next((q for q in range(gk, len(groups)) if groups[q][0][1] == start), None)
            g = groups[q] if q is not None else []
            gk = q + 1 if q is not None else gk
            tr = {
                "turn": r["turn"],
                "tier": r["tier"],
                "cpus": r["cpus"],
                "first": i == 0,
                "n_prompt": n,
                "prefill_s": t.get("prompt_ms", 0) / 1000.0,
                "read_bytes": pg["sink_bytes"] + pg["graph_bytes"] if pg else None,
                "cpu_s": pg["cpu_s"] if pg else None,
                "n_decode": t.get("predicted_n"),
                "end_pos": g[-1][1] + 1 if g else start + max(0, (t.get("predicted_n") or 0) - 1),
            }
            if g:
                L = len(a["trace"][g[0]])
                seg = [
                    [a["trace"][k][layer][0] for layer in sorted(a["trace"][k])]
                    for k in g
                    if len(a["trace"][k]) == L
                ]
                segs.append(seg)
                mine.append(seg)
                rp = m.replay(("served", d, r["tier"]), mine, r["tier"])
                (a0, a1) = rp.seg_rows[len(mine) - 1]
                (tr["model_sink_ms"], tr["model_cpu_ms"]) = m.price(
                    rp.rows[a0:a1] or rp.rows[max(0, a1 - 1) : a1], r["cpus"]
                )
                ts = r.get("tok_s") or []
                for j, k in enumerate(g):
                    if (
                        j >= 16
                        and j + 1 < len(ts)
                        and (ts[j] is not None)
                        and (ts[j + 1] is not None)
                        and (k in a["prof"])
                    ):
                        pts.append(
                            (
                                float(k[1]),
                                (ts[j + 1] - ts[j]) * 1000.0 - a["prof"][k][0] / 1000.0,
                                r["cpus"],
                            )
                        )
                win = [k for k in g[w0 - 1 : w1] if k in a["prof"]]
                rate = window_rate(ts, w0, w1)
                if win and rate:
                    tr["rate"] = rate
                    tr["pos"] = statistics.mean(k[1] for k in win)
                    tr["sink_ms"] = statistics.mean(a["prof"][k][0] for k in win) / 1000.0
                    tr["cpu_ms"] = statistics.mean(a["prof"][k][2] for k in win) / 1000.0
                    tr["out_ms"] = 1000.0 / rate - tr["sink_ms"]
            turns.append(tr)
    return (turns, segs, pts)


def fit_slope(m: Model, pts: Sequence[Point], span: float = 2000) -> None:
    """Slope of the time outside the sink against position from the served tokens (x, y,
    placement), within each placement (their levels differ), once the positions span `span`.
    """
    if not pts or max(x for (x, _, _) in pts) - min(x for (x, _, _) in pts) < span:
        return
    by: dict[str, list[tuple[float, float]]] = {}
    for x, y, p in pts:
        by.setdefault(p, []).append((x, y))
    sxy = sxx = 0.0
    for v in by.values():
        (mx, my) = (statistics.mean(x for (x, _) in v), statistics.mean(y for (_, y) in v))
        sxy += sum((x - mx) * (y - my) for (x, y) in v)
        sxx += sum((x - mx) ** 2 for (x, _) in v)
    if sxx:
        m.slope = max(0.0, sxy / sxx)


def feedback(m: Model, turns: Sequence[Record], recent: int = 1) -> Record:
    """Measured over modelled per part (the module doc); the drive's state from the last `recent`
    turns.
    """
    fb: Record = {
        "k_cpu": {},
        "out0": {},
        "k_drive": 1.0,
        "k_bytes": 1.0,
        "pf_cpu": {},
        "pf_drive": 1.0,
    }
    dec = [t for t in turns if "sink_ms" in t]
    for p in m.P:
        mine = [t for t in dec if t["cpus"] == p]
        if mine:
            fb["k_cpu"][p] = statistics.mean(t["cpu_ms"] / t["model_cpu_ms"] for t in mine)
            fb["out0"][p] = statistics.mean(t["out_ms"] - m.slope * t["pos"] for t in mine)
    if dec:
        fb["k_drive"] = statistics.mean(
            (t["sink_ms"] - t["cpu_ms"]) / (t["model_sink_ms"] - t["model_cpu_ms"])
            for t in dec[-recent:]
        )
    fb.update(
        prefill_feedback(
            m.PF,
            cast(list[ServedTurn], [t for t in turns if t["n_prompt"]]),
            lambda p: (m.route(p), m.ubatch(p)),
            recent,
        )
    )
    return fb


def expect(
    asked: Iterable[int], next_n: int, horizon: int, turns: Sequence[Record], decode: int
) -> tuple[list[int], int]:
    """The rest of a conversation whose future is not known: the next request (its new prompt
    tokens are known when it arrives), then horizon - 1 more turns of the mean new prompt
    tokens of the requests so far (`asked`) and this one. Replies: the served turns' mean
    length, `decode` before any.
    """
    ns = list(asked) + [next_n]
    ds = [t["n_decode"] for t in turns if t.get("n_decode")]
    return (
        [next_n] + [round(statistics.mean(ns))] * (horizon - 1),
        round(statistics.mean(ds)) if ds else decode,
    )


def decide(
    m: Model,
    turns: Sequence[Record],
    segs: Routing,
    cur: Sequence[Union[int, str]],
    remaining: Sequence[int],
    decode: int,
    w0: int,
    w1: int,
    start_s: float,
    keeps_history: bool = True,
    margin_frac: float = 0.05,
    margin_s: float = 15.0,
    recent: int = 1,
) -> Record:
    """Every configuration's predicted rest of the conversation; switch when the best beats
    staying by the margin. keeps_history: a restarted server restores the conversation from
    the engine's disk prompt cache; without one its first turn prefills the whole history
    again.
    """
    fb = feedback(m, turns, recent)
    ctx = turns[-1]["end_pos"] if turns else 0
    rows: list[Record] = []
    for tier, p in m.configs:
        stay = (tier, p) == tuple(cur)
        fresh = not stay and (not keeps_history) and remaining
        rem = [remaining[0] + ctx] + list(remaining[1:]) if fresh else remaining
        detail = m.predict(
            segs or m.routing,
            tier,
            p,
            0 if fresh else ctx,
            rem,
            decode,
            w0,
            w1,
            not stay,
            cast(Feedback, fb),
        )
        total = (0.0 if stay else start_s) + sum(
            t["prefill_s"] + decode / t["decode_tps"] for t in detail
        )
        rows.append(
            {
                "tier": tier,
                "cpus": p,
                "stay": stay,
                "rest_s": round(total, 1),
                "switch_cost_s": 0.0 if stay else round(start_s, 1),
                "turns": [
                    {
                        "prefill_s": round(t["prefill_s"], 1),
                        "decode_tps": round(t["decode_tps"], 3),
                        "hit_v": round(t["hit_v"], 4),
                    }
                    for t in detail
                ],
            }
        )
    rows.sort(key=lambda r: r["rest_s"])
    here = next(r for r in rows if r["stay"])
    best = rows[0]
    margin = max(margin_frac * here["rest_s"], margin_s)
    switch = not best["stay"] and here["rest_s"] - best["rest_s"] > margin
    rnd = {
        k: round(v, 4) if isinstance(v, float) else {kk: round(vv, 4) for (kk, vv) in v.items()}
        for (k, v) in fb.items()
    }
    return {
        "switch": switch,
        "to": [best["tier"], best["cpus"]] if switch else list(cur),
        "margin_s": round(margin, 1),
        "feedback": rnd,
        "slope_ms_per_ktok": round(m.slope * 1000.0, 3),
        "context": ctx,
        "remaining": remaining,
        "decode": decode,
        "candidates": rows,
    }
