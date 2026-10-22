"""F's routing replay, preserving mutable state and its original update order.

Inputs are already parsed integer expert IDs in ordered segments/tokens/layers.
This leaf has no host, file or environment dependency and does not validate or
normalize inputs. In particular, a replay checks only previous segment lengths;
it does not compare their contents. Set iteration, stable score sorting, stale
heap entries, duplicate hits and cached behind-tier queries retain F semantics.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence
from typing import Optional, TypedDict, cast

ExpertKey = tuple[int, int]
Routing = Sequence[Sequence[Sequence[Sequence[int]]]]
Rows = list[list[tuple[int, int]]]
BehindStep = tuple[int, list[int], bool]


class ReplayStats(TypedDict):
    hit_v: float
    uploads_per_token: float
    tokens: int
    hits_per_token: float


class Tier:
    cap: int
    min: float
    k: float
    ls: dict[ExpertKey, float]
    res: set[ExpertKey]
    heap: list[tuple[float, ExpertKey]]
    step: int
    admissions: int

    def __init__(self, cap: int, min_count: float, half_life: float = 16.0) -> None:
        self.cap, self.min, self.k = cap, min_count, 1.0 / half_life
        self.ls = {}  # key -> log2(count) + step / half_life: comparable across keys at any step
        self.res = set()
        self.heap = []  # (ls, key) of resident keys; entries whose ls is stale are skipped
        self.step = 0
        self.admissions = 0

    def count(self, key: ExpertKey) -> float:
        v = self.ls.get(key)
        return 0.0 if v is None else cast(float, 2.0 ** (v - self.step * self.k))

    def _bump(self, key: ExpertKey) -> None:
        v = math.log2(self.count(key) + 1.0) + self.step * self.k
        self.ls[key] = v
        if key in self.res:
            heapq.heappush(self.heap, (v, key))

    def _add(self, key: ExpertKey) -> None:
        self.res.add(key)
        heapq.heappush(self.heap, (self.ls[key], key))
        self.admissions += 1

    def access(self, layer: int, experts: Sequence[int]) -> list[bool]:
        keys = [(layer, e) for e in experts]
        for k in set(keys):
            self._bump(k)
        hits = [k in self.res for k in keys]
        if self.cap <= 0:
            return hits
        locked = {k for k, h in zip(keys, hits) if h}
        for k in sorted({k for k, h in zip(keys, hits) if not h}, key=lambda k: -self.ls[k]):
            if self.count(k) < self.min:
                continue
            if len(self.res) < self.cap:
                self._add(k)
                locked.add(k)
                continue
            held: list[tuple[float, ExpertKey]] = []
            victim: Optional[tuple[float, ExpertKey]] = None
            while self.heap:
                v, x = heapq.heappop(self.heap)
                if x not in self.res or self.ls[x] != v:
                    continue
                if x in locked:
                    held.append((v, x))
                    continue
                victim = (v, x)
                break
            for h in held:
                heapq.heappush(self.heap, h)
            if victim is None:
                continue
            if self.ls[k] < victim[0]:
                heapq.heappush(self.heap, victim)
                continue
            self.res.discard(victim[1])
            self._add(k)
            locked.add(k)
        return hits


class RamCache:
    """One RAM layer, with comparable scores log2(count) + step / 2."""

    slots: int
    ls: dict[int, float]
    res: dict[int, int]
    clock: int

    def __init__(self, slots: int) -> None:
        self.slots, self.ls, self.res, self.clock = slots, {}, {}, 0  # res: expert -> last touch

    def access(self, step: int, experts: Sequence[int]) -> int:
        """Return RAM hits among this token's tier misses at decode step `step`."""
        for e in dict.fromkeys(experts):
            c = 2.0 ** (self.ls[e] - step / 2.0) if e in self.ls else 0.0
            self.ls[e] = math.log2(c + 1.0) + step / 2.0
        hits = [e for e in experts if e in self.res]
        held = set(hits)
        for e in hits:
            self.clock += 1
            self.res[e] = self.clock
        for e in experts:
            if e in self.res or self.slots <= 0:
                continue
            if len(self.res) >= self.slots:
                free = [x for x in self.res if x not in held]
                if not free:
                    continue
                victim = min(free, key=lambda x: (self.ls[x], self.res[x]))
                if self.ls[e] < self.ls[victim]:
                    continue
                del self.res[victim]
            self.clock += 1
            self.res[e] = self.clock
            held.add(e)
        return len(hits)


class Replay:
    """Carry tier state across appended conversation segments.

    seg_rows[i] identifies the rows from segment i. behind=True also records each
    layer's tier misses at every token, for later RamCache hit estimates.
    """

    tier: Tier
    skip: int
    rows: Rows
    seg_rows: list[tuple[int, int]]
    done: list[int]
    uploads: int
    v: int
    n_exp: int
    behind: Optional[dict[int, list[BehindStep]]]
    steps: int
    _behind_hits: dict[int, float]

    def __init__(self, cap: int, min_count: float, skip: int, behind: bool = False) -> None:
        self.tier, self.skip = Tier(cap, min_count), skip
        self.rows, self.seg_rows, self.done = [], [], []
        self.uploads = self.v = self.n_exp = 0
        self.behind = {} if behind else None  # layer -> [(token step, the tier's misses, counted)]
        self.steps, self._behind_hits = 0, {}

    def advance(self, routing: Routing) -> Replay:
        if len(routing) < len(self.done) or any(len(s) != n for s, n in zip(routing, self.done)):
            raise ValueError("routing does not extend what was replayed")
        tier = self.tier
        for seg in routing[len(self.done) :]:
            start = len(self.rows)
            for ti, tok in enumerate(seg):
                a0 = tier.admissions
                row = []
                for li, ex in enumerate(tok):
                    hit = tier.access(li, ex)
                    nv = sum(hit)
                    row.append((nv, len(ex) - nv))
                    if self.behind is not None:
                        self.behind.setdefault(li, []).append(
                            (self.steps, [e for e, h in zip(ex, hit) if not h], ti >= self.skip)
                        )
                tier.step += 1
                self.steps += 1
                if ti >= self.skip:
                    self.rows.append(row)
                    self.uploads += tier.admissions - a0
                    self.v += sum(r[0] for r in row)
                    self.n_exp += sum(len(ex) for ex in tok)
            self.seg_rows.append((start, len(self.rows)))
            self.done.append(len(seg))
            self._behind_hits = {}
        return self

    def behind_hits(self, slots: int) -> float:
        """Hits per token (after skip) of a RAM cache of `slots` per layer behind the tier."""
        if slots not in self._behind_hits:
            hits = 0
            # behind=True is the caller contract. The cast is type-only: a call
            # without tracking preserves F's AttributeError rather than silently
            # treating unrecorded routes as an empty history.
            for misses in cast(dict[int, list[BehindStep]], self.behind).values():
                c = RamCache(slots)
                for step, experts, counted in misses:
                    n = c.access(step, experts)
                    hits += n if counted else 0
            self._behind_hits[slots] = hits / max(1, len(self.rows))
        return self._behind_hits[slots]

    def stats(self) -> ReplayStats:
        return {
            "hit_v": self.v / (self.n_exp or 1),
            "uploads_per_token": self.uploads / max(1, len(self.rows)),
            "tokens": len(self.rows),
            "hits_per_token": self.v / max(1, len(self.rows)),
        }


def simulate(routing: Routing, cap: int, min_count: float, skip: int) -> tuple[Rows, ReplayStats]:
    """Replay segments/tokens/layers/expert IDs, keeping the cache between segments.

    Return [(nv, nx) per layer] rows after skipping each segment's prefix, and stats.
    """
    r = Replay(cap, min_count, skip).advance(routing)
    return r.rows, r.stats()
