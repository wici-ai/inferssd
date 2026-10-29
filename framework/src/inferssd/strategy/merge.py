# SPDX-License-Identifier: Apache-2.0
"""Fill a typed settings snapshot from a strategy without overriding the user.

Precedence is CLI, then the user's settings file, then the strategy, then the
registry default. A strategy value fills only a key whose origin is still
"default"; it becomes a generated file entry with the receipt
`` source=strategy; input=<reason>``, so the launcher writes it to the settings
file it generates. Dependent values are completed so that the launcher's own
checks hold: decode participants within the decode threads, I/O CPUs outside
the strict compute CPUs, and thread counts within those CPUs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import cast

from inferssd.configuration.settings import Setting, Snapshot, parse, render
from inferssd.launch.types import Threads
from inferssd.strategy.derive import Strategy

LOCATION = "strategy"
# Server options of a strategy argument, by the spelling the strategy uses.
SERVER_OPTIONS = {"--ctx-checkpoints": ("-ctxcp", "--ctx-checkpoints", "--swa-checkpoints")}


@dataclass(frozen=True)
class Applied:
    """The merged settings and thread pools.

    ``kept`` names the strategy proposals a user value replaced; ``dropped``
    those left out because they contradict a user value. Thread pool fields
    are named ``threads.decode``, ``threads.batch`` and ``threads.cpus``.
    """

    settings: Snapshot
    threads: Threads
    kept: tuple[str, ...]
    dropped: tuple[str, ...]


def apply(strategy: Strategy, snapshot: Snapshot, threads: Threads = Threads()) -> Applied:
    """SNAPSHOT and the user's THREADS completed with STRATEGY's proposals."""
    platform = snapshot.platform
    if platform != strategy.inputs.platform:
        raise ValueError("the strategy was derived for another platform")
    kept: list[str] = []
    dropped: list[str] = []
    proposed: dict[str, Setting] = {}
    for key, choice in strategy.settings.items():
        if snapshot.at(key).origin != "default":
            kept.append(key)
            continue
        value = parse(None, LOCATION, overrides=[(key, choice.value_text)], platform=platform).at(
            key
        )
        receipt = " source=strategy; input=" + choice.reason
        proposed[key] = Setting(key, value.value, "file", LOCATION, receipt)

    # Per field, the user's thread pool value wins.
    user, own = threads, strategy.threads
    fields = {
        "decode": user.decode is None and own.decode is not None,
        "batch": user.batch is None and own.batch is not None,
        "cpus": not user.cpus and bool(own.cpus),
    }
    kept += [
        "threads." + name
        for name, given in (("decode", user.decode), ("batch", user.batch), ("cpus", user.cpus))
        if given and getattr(own, name)
    ]
    merged = Threads(
        own.decode if fields["decode"] else user.decode,
        own.batch if fields["batch"] else user.batch,
        own.cpus if fields["cpus"] else user.cpus,
    )

    def drop(name: str) -> None:
        nonlocal merged
        if name == "cpus":
            merged = replace(merged, cpus=())
        elif name == "decode":
            merged = replace(merged, decode=None)
        else:
            merged = replace(merged, batch=None)
        fields[name] = False
        dropped.append("threads." + name)

    def current(key: str) -> object:
        return proposed[key].value if key in proposed else snapshot.at(key).value

    # Thread counts within the strict CPUs.
    for name in ("decode", "batch"):
        count = getattr(merged, name)
        if not merged.cpus or count is None or count <= len(merged.cpus):
            continue
        if fields["cpus"]:
            drop("cpus")
        elif fields[name]:
            drop(name)
    # I/O CPUs outside the strict compute CPUs.
    io = set(cast(tuple[int, ...], current("cpu.io_cpus")))
    if io & set(merged.cpus):
        if "cpu.io_cpus" in proposed:
            del proposed["cpu.io_cpus"]
            dropped.append("cpu.io_cpus")
        elif fields["cpus"]:
            drop("cpus")
    # Decode participants within the decode threads (a strict CPU set sizes an unset count).
    participants = cast(int, current("cpu.decode_participants"))
    decode = merged.decode or (len(merged.cpus) if merged.cpus else None)
    if decode is not None and participants > decode:
        if "cpu.decode_participants" in proposed:
            del proposed["cpu.decode_participants"]
            dropped.append("cpu.decode_participants")
        else:
            if fields["decode"]:
                drop("decode")
            if fields["cpus"] and participants > len(merged.cpus):
                drop("cpus")

    entries = tuple(proposed.get(item.key, item) for item in snapshot.entries)
    result = replace(snapshot, entries=entries)
    # The same parser the launcher uses checks the merged values together.
    parse(render(result), "merged settings", platform=platform)
    return Applied(result, merged, tuple(kept), tuple(dropped))


def server_arguments(strategy: Strategy, extra: Sequence[str]) -> tuple[str, ...]:
    """The strategy's server arguments that the user's EXTRA arguments do not set themselves."""
    given = set()
    for argument in extra:
        name = argument.split("=", 1)[0]
        given.add(name.replace("_", "-") if name.startswith("--") else name)
    arguments: list[str] = []
    pairs = strategy.server_args
    for option, value in zip(pairs[::2], pairs[1::2]):
        if given.isdisjoint(SERVER_OPTIONS.get(option, (option,))):
            arguments += [option, value]
    return tuple(arguments)
