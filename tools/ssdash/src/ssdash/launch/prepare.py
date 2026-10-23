"""Service only the next requested observation, after delivering prior output."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping
from dataclasses import replace

from ssdash.launch.types import Fact, LaunchInput, Output, Reply, Resolution

Resolver = Callable[[LaunchInput, Mapping[Fact, Reply]], Resolution]


def prepare(
    inputs: LaunchInput,
    *,
    resolver: Resolver,
    observe: Callable[[Fact], Reply],
    emit: Callable[[Output], None],
    darwin_errors: bool = False,
) -> Resolution:
    facts: dict[Fact, Reply] = {}
    delivered: tuple[Output, ...] = ()
    result = resolver(inputs, facts)
    try:
        while True:
            if result.outputs[: len(delivered)] != delivered:
                raise RuntimeError("Resolver changed an already delivered output prefix")
            for event in result.outputs[len(delivered) :]:
                emit(event)
                delivered += (event,)
            if result.needed is None:
                return result
            facts[result.needed] = observe(result.needed)
            result = resolver(inputs, facts)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        if not darwin_errors:
            raise
        event = Output("stderr", str(error) + "\n")
        emit(event)
        return replace(result, outputs=delivered + (event,), needed=None, code=2)
