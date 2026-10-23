"""Explicit offline planner arguments and output port."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class PlanArguments:
    prompts: str
    decode: int
    window: str
    known: bool
    pending: str
    out: str


class Emit(Protocol):
    def __call__(self, *values: object) -> None: ...
