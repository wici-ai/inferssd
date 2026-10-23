"""Tier observations from existing ubatch run logs; no probe execution."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from typing import Optional, cast

from inferssd.bootstrap.ports import Emit, OpenText

TIER = re.compile(
    r"\[kimi_vram_tier\] reason=allocated .*? slots=(\d+)"
    r"|\[kimi_vram_elastic\] event=decode_grow .*? after_slots=(\d+)"
    r"|\[kimi_vram_elastic\] event=watermark_grow .*? slots=(\d+)"
)


def reached(lines: Iterable[str]) -> tuple[Optional[int], Optional[int]]:
    """First decode growth (else first allotment), and maximum held slots.

    Return (None, None) when no allotted tier is observed.
    """
    first: Optional[int]
    held: Optional[int]
    first = held = None
    grown = False
    for line in lines:
        m = TIER.search(line)
        if not m:
            continue
        alloc, grow, mark = (int(g) if g else None for g in m.groups())
        n = alloc or grow or mark
        held = n if held is None else max(held, cast(int, n))
        if grow is not None and not grown:
            first, grown = grow, True
        elif alloc is not None and first is None:
            first = alloc
    return first, held


def run_ubprobe(
    argv: Sequence[str], *, open_file: OpenText, open_gzip: OpenText, emit: Emit
) -> None:
    out, ubs = argv[1], argv[2:]
    res: dict[str, Optional[dict[str, Optional[int]]]] = {}
    for ub in ubs:
        try:
            run = open_file(f"{out}/ubprobe-{ub}.out").readline().strip()
            first, held = reached(open_gzip(f"{run}/server.log.gz", "rt", errors="replace"))
            res[ub] = {"first": first, "held": held} if held is not None else None
        except OSError:
            res[ub] = None
    emit(json.dumps(res))
