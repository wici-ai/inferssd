"""Strict v1 JSONL consumption; unknown event kinds remain inspectable records."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from inferssd.configuration.settings import schema, typed


@dataclass(frozen=True)
class Event:
    version: int
    name: str
    sequence: int
    payload: Mapping[str, Any]


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field " + key)
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON number " + value)


def finite_number(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite JSON number " + value)
    return number


def finite_integer(value: str) -> int:
    number = int(value)
    try:
        finite_number(str(number))
    except OverflowError as error:
        raise ValueError("out-of-range JSON number") from error
    return number


def read_events(lines: Iterable[str]) -> Iterable[Event]:
    for expected, line in enumerate(lines):
        try:
            value: object = json.loads(
                line,
                object_pairs_hook=unique_object,
                parse_constant=reject_constant,
                parse_float=finite_number,
                parse_int=finite_integer,
            )
            if not isinstance(value, dict):
                raise ValueError("event must be an object")
            if type(value.get("v")) is not int or value["v"] != 1:
                raise ValueError("unsupported telemetry version")
            if type(value.get("seq")) is not int or value["seq"] != expected:
                raise ValueError("missing, duplicate or out-of-order telemetry sequence")
            name = value.get("ev")
            if (
                not isinstance(name, str)
                or re.fullmatch(r"ssd\.[a-z0-9_]+(?:\.[a-z0-9_]+)*", name) is None
            ):
                raise ValueError("invalid telemetry event name")
            yield Event(
                1, name, expected, {k: v for k, v in value.items() if k not in ("v", "ev", "seq")}
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"telemetry line {expected + 1}: {error}") from error


def effective_settings(events: Iterable[Event]) -> Mapping[str, Any]:
    selected = [event for event in events if event.name == "ssd.settings"]
    if len(selected) != 1:
        raise ValueError("expected exactly one ssd.settings event per runtime")
    payload = selected[0].payload
    if not isinstance(payload.get("values"), dict) or not isinstance(payload.get("sources"), dict):
        raise ValueError("ssd.settings requires values and source receipts")
    properties = schema()["properties"]
    values, sources = payload["values"], payload["sources"]
    if set(values) != set(properties) or set(sources) != set(properties):
        raise ValueError("ssd.settings must contain exactly the registered settings/receipts")
    for key, prop in properties.items():
        value, receipt = values[key], sources[key]
        if not isinstance(receipt, dict) or set(receipt) != {"origin", "location", "receipt"}:
            raise ValueError("invalid settings receipt for " + key)
        if receipt["origin"] not in ("cli", "file", "default") or any(
            not isinstance(item, str) for item in receipt.values()
        ):
            raise ValueError("invalid settings source for " + key)
        kind = prop["kind"]
        if kind == "integer" or kind == "budget" and value != "auto":
            if type(value) is not int:
                raise ValueError("expected integer setting " + key)
            literal = str(value)
        elif kind == "budget":
            literal = "auto"
        elif kind == "cpus":
            if not isinstance(value, list) or any(type(item) is not int for item in value):
                raise ValueError("expected CPU list setting " + key)
            literal = ",".join(map(str, value))
        elif kind == "costs":
            if not isinstance(value, dict) or any(
                type(v) not in (int, float) for v in value.values()
            ):
                raise ValueError("expected cost table setting " + key)
            literal = ",".join(f"{k}={v}" for k, v in value.items())
        else:
            literal = json.dumps(value, ensure_ascii=False)
        typed(prop, literal)
    return payload
