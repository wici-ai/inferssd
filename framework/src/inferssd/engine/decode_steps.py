# SPDX-License-Identifier: Apache-2.0
"""The engine's measurement of a model's decode steps, and the measurements this machine keeps.

``inferssd-server <a launch's arguments> --ssd-decode-probe`` loads the model and its context as
the launch would serve them (without the VRAM expert tier and read-ahead), binds no port, times
target decode steps of 1 to n_max + 1 rows and a step of the MTP head the launch drafts with,
prints one JSON object and exits. Its dense times are what the strategy cannot price from the
model's bytes. A probe takes minutes, as loading the model does, so its measurement is kept beside
the machine characteristics and reused while everything it depends on is the same: the server
build, the characteristics, the model's and the head's files and the launch's arguments.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union, cast

from inferssd.contracts import ContractError
from inferssd.engine.model import Run
from inferssd.machine.characteristics import (
    COUNT,
    NUMBER,
    POSITIVE,
    TEXT,
    array,
    integer,
    nullable,
    record,
    utc,
)
from inferssd.machine.store import LockBusy, default_path, exclusive, replace_text

VERSION = 1
FLAG = "--ssd-decode-probe"
# The model and its context load as for serving before the steps are timed: minutes from a cold
# drive.
TIMEOUT_S = 1800.0
FILENAME = "decode-steps.json"
SCHEMA = "inferssd-decode-steps/1"
# The options that name what is a run's own: its port and its telemetry file. Its settings file
# lies in the run directory too; the digest of its content stands for it.
VOLATILE = ("--port", "--ssd-telemetry")

STEP = {"ms": NUMBER, "sink_ms": NUMBER, "dense_ms": NUMBER}
MEASUREMENT = record(
    {
        "version": integer(VERSION),
        "steps": POSITIVE,
        "rows": array(record({"rows": POSITIVE, **STEP}), minimum=1),
        "draft": nullable(record(STEP)),
    }
)


def _file(value: object, field: str) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ContractError(f"{field} must be [a path, its bytes]")
    TEXT(value[0], field + "[0]")
    COUNT(value[1], field + "[1]")


ENTRY = record(
    {
        "key": TEXT,
        "server_sha256": TEXT,
        "characteristics_sha256": TEXT,
        "model": array(_file, minimum=1),
        "head": _file,
        "arguments": array(TEXT, minimum=1),
        "measured_utc": utc,
        "measurement": MEASUREMENT,
    }
)


def validate(value: object) -> dict[str, Any]:
    """VALUE as the probe's measurement; errors name the offending field."""
    MEASUREMENT(value, "decode_probe")
    measured = cast(dict[str, Any], value)
    if measured["version"] != VERSION:
        raise ContractError(f"decode_probe.version must be {VERSION}")
    for index, row in enumerate(measured["rows"]):
        if row["rows"] != index + 1:
            raise ContractError(f"decode_probe.rows[{index}].rows must be {index + 1}")
    return measured


def measure(argv: Sequence[str], run: Run) -> Union[dict[str, Any], str]:
    """The measurement the server command ARGV prints, or why there is none."""
    try:
        done = run(argv)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        return f"the server did not run: {error}"
    if done.returncode != 0:
        said = [line for line in done.stderr.splitlines() if line.strip()]
        errors = [line[len("error: ") :] for line in said if line.startswith("error: ")]
        last = errors[-1] if errors else said[-1] if said else "no reason given"
        return f"the server exited with status {done.returncode}: {last}"
    # The server's own log lines may come first; the measurement is the last line.
    lines = [line for line in done.stdout.splitlines() if line.strip()]
    try:
        return validate(json.loads(lines[-1] if lines else ""))
    except ValueError as error:
        return f"the server printed no measurement: {error}"


@dataclass(frozen=True)
class Request:
    """A decode probe a launch asks for: its server command and what its measurement holds for."""

    characteristics_sha256: str  # the machine characteristics the launch was derived from
    settings_sha256: str  # the content of the settings file the command names
    model: tuple[str, ...]  # the model's files
    head: str  # the MTP head file the command drafts with
    argv: tuple[str, ...]  # the server command, FLAG among its arguments

    def arguments(self) -> tuple[str, ...]:
        """The request as a fact's arguments; parse reads them back."""
        return (
            self.characteristics_sha256,
            self.settings_sha256,
            self.head,
            str(len(self.model)),
            *self.model,
            *self.argv,
        )

    @staticmethod
    def parse(arguments: Sequence[str]) -> Request:
        count = int(arguments[3])
        return Request(
            arguments[0],
            arguments[1],
            tuple(arguments[4 : 4 + count]),
            arguments[2],
            tuple(arguments[4 + count :]),
        )

    def stable_arguments(self) -> list[str]:
        """The command without what is the run's own: VOLATILE options left out, the settings
        file named by the digest of its content."""
        stable: list[str] = []
        argv = list(self.argv)
        while argv:
            option = argv.pop(0)
            if option in VOLATILE and argv:
                argv.pop(0)
            elif option == "--ssd-settings" and argv:
                argv.pop(0)
                stable += [option, "sha256:" + self.settings_sha256]
            else:
                stable.append(option)
        return stable


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(request: Request) -> dict[str, Any]:
    """What a measurement of REQUEST holds for, keyed by the digest of it all."""
    entry: dict[str, Any] = {
        "server_sha256": file_sha256(request.argv[0]),
        "characteristics_sha256": request.characteristics_sha256,
        "model": [[path, os.path.getsize(path)] for path in request.model],
        "head": [request.head, os.path.getsize(request.head)],
        "arguments": request.stable_arguments(),
    }
    key = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
    return {"key": key, **entry}


def store_path(platform: str, environ: Mapping[str, str], home: Optional[Path] = None) -> Path:
    """The per-user file of the kept measurements, beside the machine characteristics."""
    return default_path(platform, environ, home).with_name(FILENAME)


def kept(path: Path) -> list[dict[str, Any]]:
    """The measurements kept in PATH; none when it is missing or unreadable, and an entry that
    does not read as one is left out."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        return []
    listed = data.get("entries")
    entries = []
    for entry in listed if isinstance(listed, list) else []:
        try:
            ENTRY(entry, "entry")
            validate(entry["measurement"])
        except ContractError:
            continue
        entries.append(cast(dict[str, Any], entry))
    return entries


def cached(request: Request, path: Path) -> Optional[dict[str, Any]]:
    """The measurement of REQUEST kept in PATH, None when there is none."""
    key = identity(request)["key"]
    return next((entry["measurement"] for entry in kept(path) if entry["key"] == key), None)


def probe(request: Request, path: Path, run: Run) -> Union[dict[str, Any], str]:
    """REQUEST measured now with RUN and kept in PATH, or why the probe measured nothing."""
    entry = identity(request)
    measured = measure(request.argv, run)
    if isinstance(measured, str):
        return measured
    entry["measured_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry["measurement"] = measured
    # A measurement that cannot be kept still decides this launch; the next one measures again.
    with contextlib.suppress(OSError, LockBusy), exclusive(path, 10.0):
        entries = [old for old in kept(path) if old["key"] != entry["key"]] + [entry]
        replace_text(path, json.dumps({"schema": SCHEMA, "entries": entries}, indent=2) + "\n")
    return measured


def dense_ms(
    entries: Sequence[Mapping[str, Any]],
    server_sha256: str,
    characteristics_sha256: str,
    model: Sequence[str],
) -> Optional[float]:
    """The dense time of a one-row step of MODEL (its files, by name) measured with this server
    on this machine, by any launch arguments: the latest such measurement's, None without one."""
    names = sorted(os.path.basename(path) for path in model)
    matching = [
        entry
        for entry in entries
        if entry["server_sha256"] == server_sha256
        and entry["characteristics_sha256"] == characteristics_sha256
        and sorted(os.path.basename(path) for path, _ in entry["model"]) == names
    ]
    if not matching:
        return None
    latest = max(matching, key=lambda entry: cast(str, entry["measured_utc"]))
    return cast(float, latest["measurement"]["rows"][0]["dense_ms"])
