# SPDX-License-Identifier: Apache-2.0
"""Conservative launch-time RAM admission; runtime guards remain authoritative."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from inferssd.platform.resources import command

GIB = 1024**3


def available_memory() -> int:
    """Use the same available-memory estimate as the owned runtime guard."""
    if sys.platform == "linux":
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    elif sys.platform == "darwin":
        vm = command(["vm_stat"])
        size = re.search(r"page size of (\d+) bytes", vm)
        pages = {name.strip(): int(n) for name, n in re.findall(r"^([^:\n]+):\s*(\d+)\.", vm, re.M)}
        if size is not None and {"Pages free", "Pages inactive"} <= pages.keys():
            return (pages["Pages free"] + pages["Pages inactive"]) * int(size[1])
    raise ValueError("cannot read available RAM; provide --runtime-policy")


def aligned_budget(budget: int) -> int:
    """Linux rounds MemoryMax down to pages; never relax a requested cap."""
    page = int(os.sysconf("SC_PAGE_SIZE")) if sys.platform == "linux" else 1
    return budget // page * page


@dataclass(frozen=True)
class Admission:
    total_bytes: int
    available_bytes: int
    host_reserve_bytes: int
    startup_margin_bytes: int
    safe_budget_bytes: int
    requested_bytes: Optional[int]
    memory_budget_bytes: int


def admit(total: int, available: int, requested: Optional[int] = None) -> Admission:
    """Leave host reserve plus uncommitted room for startup and sampling races.

    This is a snapshot, not a reservation. Other processes can still consume RAM;
    cgroup/host-availability checks must continue throughout the owned session.
    """
    if total <= 0 or available < 0 or available > total:
        raise ValueError("invalid physical/available RAM estimate")
    reserve = max(2 * GIB, total // 5)
    margin = max(GIB, total // 20)
    ceiling = aligned_budget(available - reserve - margin)
    if ceiling <= 0:
        raise ValueError(
            f"not enough available RAM: {available / GIB:.2f} GiB available; "
            f"need {reserve / GIB:.2f} GiB host reserve + "
            f"{margin / GIB:.2f} GiB startup margin before allocating server RAM"
        )
    budget = ceiling if requested is None else aligned_budget(requested)
    if budget <= 0 or budget > ceiling:
        raise ValueError(
            f"server RAM budget must be positive and <= {ceiling / GIB:.3f} GiB "
            f"({ceiling} bytes) at present: {available / GIB:.2f} GiB available, "
            f"{reserve / GIB:.2f} GiB host reserve, "
            f"{margin / GIB:.2f} GiB startup margin; "
            "omit --memory-gib to select automatically or free memory and retry"
        )
    return Admission(total, available, reserve, margin, ceiling, requested, budget)
