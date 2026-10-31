# SPDX-License-Identifier: Apache-2.0
"""The ``inferssd profile`` command: measure this machine once and keep the result."""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Optional

from inferssd.contracts import ContractError
from inferssd.machine.characteristics import (
    BASE_SECTIONS,
    STORAGE_PREFIX,
    Characteristics,
    Section,
)
from inferssd.machine.identity import Host, NativeHost
from inferssd.machine.probes import Probes, Runner, locate, run_process
from inferssd.machine.profile import Clock, Progress, profile, stderr_progress
from inferssd.machine.store import default_path, exclusive, load_current, save

GIB = float(1 << 30)


def add_command(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    command = commands.add_parser(
        "profile",
        help="measure this machine's CPU, memory, GPUs and model drives (read-only)",
        description="Measure this machine once and save its characteristics. Sections that "
        "are still valid are reused; drive measurements only read existing files.",
    )
    command.add_argument(
        "--model",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help="model file or directory whose drive to measure (repeatable); "
        "a split GGUF counts as all its shards",
    )
    probes = command.add_mutually_exclusive_group()
    probes.add_argument(
        "--resources", type=Path, help="inferssd-engine 1 descriptor naming the probes"
    )
    probes.add_argument(
        "--build", type=Path, help="engine build directory holding bin/inferssd-*probe"
    )
    command.add_argument("--reprofile", action="store_true", help="measure every section again")
    command.add_argument(
        "--output", type=Path, help="characteristics file (default: the per-user state directory)"
    )
    command.add_argument(
        "--show", action="store_true", help="print the saved profile; measure nothing"
    )


def _gib(value: float) -> str:
    return f"{value / GIB:.1f} GiB"


def _rate(value: float) -> str:
    return f"{value / GIB:.2f} GiB/s"


def _size(value: int) -> str:
    return f"{value >> 20} MiB" if value >= 1 << 20 and value % (1 << 20) == 0 else f"{value} B"


def _facts(value: Characteristics, name: str) -> str:
    if name == "cpu":
        cpu = value["cpu"]
        groups = ", ".join(f"{len(row['cpus'])} {row['kind']}" for row in cpu["clusters"])
        isa = "; ISA " + " ".join(cpu["isa"]) if cpu["isa"] else ""
        return f"{len(cpu['logical'])} logical CPUs ({groups}){isa}"
    if name == "memory":
        memory = value["memory"]
        dram = ", ".join(
            f"{_rate(row['Bps'])} with {row['threads']} thread{'' if row['threads'] == 1 else 's'}"
            for row in memory["dram_read_Bps"]
        )
        unified = "unified, " if memory["unified"] else ""
        return f"{_gib(memory['total_bytes'])}, {unified}{memory['page_bytes']} B pages" + (
            f"; DRAM read {dram}" if dram else ""
        )
    if name == "compute":
        rows = value["compute"]["cpu"]
        formats = sorted({row["format"] for row in rows})
        threads = sorted({row["threads"] for row in rows})
        return f"{len(formats)} formats at {','.join(map(str, threads))} threads" if rows else ""
    if name == "gpus":
        return "; ".join(
            f"{gpu['name']} {_gib(gpu['vram_total_bytes'])}, host to device "
            + _rate(max((row["Bps"] for row in gpu["h2d"]["pinned"]), default=0.0))
            for gpu in value["gpus"]
        )
    key = name[len(STORAGE_PREFIX) :]
    for entry in value["storage"]:
        if entry["drive_key"] == key:
            best = max(entry["random_read"], key=lambda row: row["Bps"])
            knees = ", ".join(f"QD {row['qd']} at {_size(row['bytes'])}" for row in entry["knee"])
            return (
                f"{entry['model']} ({entry['bus']}, {entry['reader']}), "
                f"{entry['measured_files']} files; best {_rate(best['Bps'])} at "
                f"{_size(best['bytes'])} QD {best['qd']}; knee {knees}"
            )
    return ""


def summary(value: Characteristics, path: Path) -> list[str]:
    """One header line, then one line per section."""
    hard = value["identity"]["hard"]
    gpus = ", ".join(f"{gpu['name']} ({gpu['backend']})" for gpu in hard["gpus"]) or "none"
    lines = [
        f"machine profile {path} (written {value['created_utc']})",
        f"identity: {hard['platform']} {hard['arch']}, {hard['cpu_brand']}, "
        f"{hard['logical_cpus']} logical CPUs, {_gib(hard['ram_bytes'])} RAM, GPUs: {gpus}",
    ]
    order = {name: index for index, name in enumerate(BASE_SECTIONS)}
    for name, section in sorted(
        value["sections"].items(), key=lambda item: (order.get(item[0], len(order)), item[0])
    ):
        notes = "; ".join(filter(None, (_facts(value, name), section["detail"])))
        lines.append(
            f"{name}: {section['status']} ({section['duration_s']:.1f} s, "
            f"{section['measured_utc']}){': ' + notes if notes else ''}"
        )
    return lines


def _changed(value: Characteristics, previous: Optional[Characteristics]) -> dict[str, Section]:
    old = previous["sections"] if previous is not None else {}
    # Reused sections are the saved records themselves; measured ones are new.
    return {name: rec for name, rec in value["sections"].items() if old.get(name) is not rec}


def run(
    args: argparse.Namespace,
    *,
    host: Optional[Host] = None,
    runner: Optional[Runner] = None,
    environ: Mapping[str, str] = os.environ,
    home: Optional[Path] = None,
    clock: Clock = time.time,
    progress: Progress = stderr_progress,
    emit: Callable[[str], None] = print,
) -> int:
    host = host or NativeHost()
    path = args.output or default_path(host.platform(), environ, home)
    if args.show:
        if args.model or args.reprofile or args.resources or args.build:
            raise ValueError("--show measures nothing; it accepts only --output")
        saved = load_current(path)
        if saved is None:
            raise ValueError(f"no machine profile at {path}; run inferssd profile first")
        for line in summary(saved, path):
            emit(line)
        return 0
    suffix = ".exe" if host.platform() == "windows" else ""
    resources = args.resources
    if resources is None and args.build is None:
        installed = environ.get("INFERSSD_RESOURCES")
        if installed:
            resources = Path(installed)
    probes = Probes(
        tuple(locate(resources, args.build, suffix=suffix).items()), runner or run_process
    )
    with exclusive(path):
        try:
            previous = load_current(path)
        except (ContractError, UnicodeDecodeError) as error:
            progress(f"ignoring the unreadable saved profile {path}: {error}")
            previous = None
        value = profile(
            host,
            probes,
            [str(model) for model in args.model],
            previous,
            args.reprofile,
            clock=clock,
            progress=progress,
        )
        if value is not previous:
            save(value, path)
    for line in summary(value, path):
        emit(line)
    changed = _changed(value, previous)
    partial = [name for name, rec in changed.items() if rec["status"] == "partial"]
    failed = [name for name, rec in changed.items() if rec["status"] == "failed"]
    if partial:
        emit(f"{len(partial)} section(s) partial: {', '.join(partial)}")
    if failed:
        emit(f"{len(failed)} section(s) failed: {', '.join(failed)}")
    return 1 if failed else 0
