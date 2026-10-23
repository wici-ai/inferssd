"""Typed production selection from measured model fields, without settings.env."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import replace
from typing import Optional

from inferssd.configuration.settings import Setting, Snapshot, parse, render
from inferssd.model.machine import Machine
from inferssd.plan import _prefill as prefill


class SelectionConflict(ValueError):
    """An explicit CLI setting excludes a measured adaptive selection."""


def selection(
    machine: Machine,
    tier: int,
    placement: str,
    base: Snapshot,
    host_threads: Callable[[], Optional[int]],
    *,
    native_ubatch: Optional[int] = None,
) -> Snapshot:
    if base.platform != "linux":
        raise ValueError("Darwin online adaptive placement M2 remains unimplemented")
    if placement not in machine.P or tier < 0 or tier > machine.caps[placement]:
        raise ValueError("selection is outside measured placement/tier capacity")
    chosen: dict[str, tuple[str, str]] = {}

    def choose(key: str, value: object, receipt: str) -> None:
        chosen[key] = (str(value), " source=planner; input=" + receipt)

    name = placement.partition("@")[0]
    groups: list[list[int]] = []
    coresets = machine.dev.get("coresets", {})
    for index in range(1, len(coresets) + 1):
        if "g" + str(index) not in coresets:
            break
        groups.append(list(coresets["g" + str(index)]["cpus"]))
    if name not in ("all", "pin", "split"):
        raise ValueError("unimplemented typed CPU placement " + name)
    total = 0 if name == "pin" else int(machine.dev.get("default_threads") or host_threads() or 0)
    if name == "all":
        if total < 2:
            raise ValueError("all placement requires a measured or explicit host thread count")
        compute, io, participants = [], [], 0
    else:
        if len(groups) < 2 or len(groups[0]) < 3:
            raise ValueError("pin/split requires measured CPU groups")
        io = [groups[0][-1]]
        if name == "pin":
            compute = groups[0][:-1]
            total = len(compute) + 1
            participants = len(compute)
        else:
            compute = [cpu for group in groups for cpu in group][:total]
            compute = [cpu for cpu in compute if cpu not in io]
            if len(compute) != total - 1:
                raise ValueError("split placement has insufficient measured CPU IDs")
            participants = len(groups[0]) - 1
    choose("cpu.threads", total, "device.default_threads and measured placement " + placement)
    choose(
        "cpu.compute_cpus",
        json.dumps(",".join(map(str, compute))),
        "measured compute CPU IDs " + placement,
    )
    choose("cpu.io_cpus", json.dumps(",".join(map(str, io))), "measured I/O CPU ID " + placement)
    choose("cpu.decode_participants", participants, "measured fast CPU group " + placement)
    choose(
        "vram.max_slots",
        tier,
        "requested tier within measured capacity " + str(machine.caps[placement]),
    )
    if tier == 0:
        choose("vram.budget", 0, "CPU-only selection")
    else:
        reference = machine.P[machine.ref]
        slot, original = reference.get("tier_slot_bytes"), reference.get("tier_base_bytes")
        if slot and original and not (machine.dev.get("gpu") or {}).get("integrated"):
            if tier * slot < original:
                choose("vram.budget", int(tier * slot), "measured slot bytes and selected tier")
    record = machine.P[placement]
    depth = record.get("pf_k")
    if depth is not None:
        choose("prefetch.depth", int(depth), "measured placement pf_k " + placement)
    limit = (
        (machine.PF.get("ubatch_p") or {}).get(name)
        or machine.PF.get("ubatch")
        or (machine.PF["memory"].get(placement) or {}).get("n_ubatch")
    )
    if limit:
        if native_ubatch is not None:
            limit = min(int(limit), native_ubatch)
        ubatch = placement.partition("@")[2]
        cut = (
            int(ubatch)
            if ubatch
            else prefill.graph_pass_max(machine.PF, placement, int(limit)) or int(limit)
        )
        threshold = prefill.threshold(machine.PF, placement, None, cut, 0)
        choose(
            "prefill.cpu_max_tokens",
            (1 << 32) - 1 if threshold is None or threshold >= cut else int(threshold),
            "measured graph/sink crossover " + placement,
        )
    costs: dict[str, float] = {}
    lanes = machine.dev.get("lanes") or {}
    for lane in ("cpu", "gpu", "gpu_host"):
        line = lanes.get(lane)
        if isinstance(line, dict):
            for source, suffix in (("fixed_us", "fixed_us"), ("per_route_us", "route_us")):
                if source in line:
                    value = float(line[source])
                    if not math.isfinite(value) or value < 0:
                        raise ValueError("invalid measured lane cost")
                    costs[lane + "_" + suffix] = value
    link = lanes.get("copy_gbps")
    if link:
        costs["link_gbps"] = float(link)
    if costs:
        choose(
            "place.costs",
            json.dumps(",".join(key + "=" + format(value, ".17g") for key, value in costs.items())),
            "measured device CPU/GPU/copy lanes",
        )
    margins = []
    for record in machine.P.values():
        ledger = record.get("ledger")
        if ledger and ledger.get("unified") and ledger.get("margin_updates"):
            margins.append(int(ledger["margin"]))
    if margins:
        choose(
            "host.headroom_mib",
            -(-max(margins) // (1 << 20)),
            "measured bootstrap host ledger margins",
        )
    # Validate all dependencies using the same v1 parser, then restore the
    # original untouched receipts and label selected fields as generated file
    # inputs. Native CLI provenance remains reserved for the user's CLI.
    checked = parse(
        render(base),
        "base snapshot",
        overrides=[(k, v[0]) for k, v in chosen.items()],
        platform="linux",
    )
    entries = []
    for item in base.entries:
        if item.key in chosen:
            validated = checked.at(item.key)
            if item.origin == "cli":
                if item.value != validated.value:
                    raise SelectionConflict(
                        "explicit override excludes measured selection: " + item.key
                    )
                entries.append(item)
                continue
            entries.append(
                Setting(
                    item.key, validated.value, "file", "planner " + placement, chosen[item.key][1]
                )
            )
        else:
            entries.append(item)
    return replace(base, entries=tuple(entries))
