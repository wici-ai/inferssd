"""Immutable launch inputs, observations and ordered output events."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional, cast


@dataclass(frozen=True)
class LaunchInput:
    repo: str
    environment: tuple[tuple[str, str], ...]
    extra: tuple[str, ...] = ()
    profile: str = "muse"
    mode: str = "server"


@dataclass(frozen=True)
class Fact:
    name: str
    arguments: tuple[str, ...] = ()
    occurrence: int = 0


@dataclass(frozen=True)
class Reply:
    value: object = None
    stdout: str = ""
    stderr: str = ""
    code: int = 0


@dataclass(frozen=True)
class Output:
    channel: str
    text: str


@dataclass(frozen=True)
class Resolution:
    argv: tuple[str, ...]
    assigned_environment: tuple[tuple[str, str], ...]
    effective_environment: tuple[tuple[str, str], ...]
    scope: tuple[str, ...]
    outputs: tuple[Output, ...]
    observations: tuple[Fact, ...]
    needed: Optional[Fact]
    code: Optional[int]


class AwaitFact(Exception):
    def __init__(self, fact: Fact):
        self.fact = fact


class Finished(Exception):
    def __init__(self, code: int):
        self.code = code


class State:
    def __init__(self, inputs: LaunchInput, facts: Mapping[Fact, Reply]):
        self.inputs = inputs
        self.facts = facts
        self.env = dict(inputs.environment)
        self.outputs: list[Output] = []
        self.observations: list[Fact] = []
        self.argv: list[str] = []
        self.assigned: list[tuple[str, str]] = []
        self.scope: list[str] = []

    def ask(self, name: str, *arguments: str) -> Reply:
        occurrence = sum(f.name == name and f.arguments == arguments for f in self.observations)
        fact = Fact(name, arguments, occurrence)
        self.observations.append(fact)
        if fact not in self.facts:
            raise AwaitFact(fact)
        return self.facts[fact]

    def value(self, name: str, *arguments: str) -> object:
        return self.ask(name, *arguments).value

    def text(self, name: str, *arguments: str) -> str:
        return cast(str, self.value(name, *arguments))

    def emit(self, text: str, channel: str = "stdout") -> None:
        self.outputs.append(Output(channel, text))

    def error(self, text: str, code: int = 1, *, append_status: bool = False) -> None:
        self.emit(text + (f" {code}" if append_status else "") + "\n", "stderr")
        raise Finished(code)

    def metadata(self, name: str, *arguments: str) -> Reply:
        reply = self.ask(name, *arguments)
        if reply.stderr:
            self.emit(reply.stderr, "stderr")
        return reply

    def finish(self, code: Optional[int], needed: Optional[Fact] = None) -> Resolution:
        effective = dict(self.env)
        effective.update(self.assigned)
        return Resolution(
            tuple(self.argv),
            tuple(self.assigned),
            tuple(effective.items()),
            tuple(self.scope),
            tuple(self.outputs),
            tuple(self.observations),
            needed,
            code,
        )
