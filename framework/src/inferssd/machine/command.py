# SPDX-License-Identifier: Apache-2.0
"""The ``inferssd profile`` command: measure this machine once and keep the result."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Optional

from inferssd.app import catalog
from inferssd.contracts import ContractError
from inferssd.engine import model as engine_model
from inferssd.launch.resources import executable, installed_resources, read_resources
from inferssd.machine.characteristics import (
    BASE_SECTIONS,
    STORAGE_PREFIX,
    Characteristics,
    Section,
)
from inferssd.machine.decode_probe import measure_decode_compute
from inferssd.machine.identity import Host, NativeHost, UnsupportedPlatform
from inferssd.machine.probes import Probes, Runner, locate, run_process
from inferssd.machine.profile import (
    Clock,
    Progress,
    model_drives,
    models_drive,
    profile,
    stderr_progress,
    survey,
)
from inferssd.machine.store import default_path, exclusive, load_current, save
from inferssd.strategy import calibration
from inferssd.strategy.estimate import Estimate, decode_estimate, dense_decode_estimate
from inferssd.strategy.inputs import RuntimeFacts, facts_from_engine, read_runtime
from inferssd.strategy.weights import DenseWeights, dense_weights
from inferssd.telemetry.routing import last_request_hits

GIB = float(1 << 30)


def add_command(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    command = commands.add_parser(
        "profile",
        help="measure this machine's CPU, memory, GPUs and model drives; estimate model speeds",
        description="Measure this machine once and save its characteristics. Sections that "
        "are still valid are reused; drive measurements read existing files, or a test file "
        "they write and remove in a directory without one.",
    )
    command.add_argument(
        "--model",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help="model file or directory whose drive to measure (repeatable); "
        "a split GGUF counts as all its shards; a directory without a file of 1 GiB "
        "is measured with a 4 GiB test file, removed after; the speed estimates "
        "take the models to be on the drive that holds most of the files given",
    )
    probes = command.add_mutually_exclusive_group()
    probes.add_argument(
        "--resources", type=Path, help="inferssd-engine 1 descriptor naming the probes"
    )
    probes.add_argument(
        "--build", type=Path, help="engine build directory holding bin/inferssd-*probe"
    )
    command.add_argument("--memory-gib", type=float, help="host memory budget for estimation only")
    target = command.add_mutually_exclusive_group()
    target.add_argument(
        "--estimate-gguf",
        type=Path,
        metavar="FILE",
        help="estimate this GGUF using the engine's header reader",
    )
    target.add_argument(
        "--estimate-facts",
        type=Path,
        metavar="JSON",
        help="estimate exported --ssd-model-facts on this machine",
    )
    command.add_argument(
        "--shard",
        "--ple",
        dest="shards",
        action="append",
        default=[],
        metavar="FILE,...",
        help="additional GGUF files for --estimate-gguf",
    )
    command.add_argument(
        "--routing-events",
        type=Path,
        metavar="NATIVE-EVENTS.jsonl",
        help="use cache hits from the last request in a representative single-stream run",
    )
    calibration_args = command.add_mutually_exclusive_group()
    calibration_args.add_argument(
        "--calibrate-from",
        type=Path,
        metavar="RESPONSE.json",
        help="derive a machine calibration from a resident reference model's response timings",
    )
    calibration_args.add_argument(
        "--calibration",
        type=Path,
        metavar="PROFILE.json",
        help="apply an existing machine calibration to the requested GGUF",
    )
    command.add_argument(
        "--host-work-mib",
        type=float,
        default=0,
        help="additional known host working memory for estimation",
    )
    command.add_argument(
        "--gpu-work-mib",
        type=float,
        default=0,
        help="known GPU KV/graph working memory for estimation",
    )
    command.add_argument(
        "--estimate-compute",
        action="store_true",
        help="measure synthetic decode matrices on CUDA or ROCm GPUs; never runs a model",
    )
    command.add_argument("--reprofile", action="store_true", help="measure every section again")
    command.add_argument(
        "--output", type=Path, help="characteristics file (default: the per-user state directory)"
    )
    command.add_argument(
        "--show", action="store_true", help="print the saved profile; measure nothing"
    )
    command.add_argument(
        "--json",
        action="store_true",
        help="print the profile and the supported models' decode speed estimates "
        "as one JSON object",
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


def estimates(
    value: Characteristics,
    drive_key: Optional[str],
    platform: str,
    runtime: Callable[[], RuntimeFacts],
    *,
    host_work_bytes: int = 0,
    gpu_work_bytes: int = 0,
    target: Optional[catalog.Model] = None,
    weights: Optional[DenseWeights] = None,
    observed_hits: Optional[tuple[float, float]] = None,
) -> list[tuple[catalog.Model, Estimate]]:
    """Each supported model's decode speed estimate with its shards on DRIVE_KEY."""
    reason = "pass --model with the directory that holds the models"
    models = (target,) if target else catalog.MODELS
    if drive_key is not None or target is not None:
        try:
            facts = runtime()
        except UnsupportedPlatform as error:
            reason = str(error)
        else:
            return [
                (
                    model,
                    decode_estimate(
                        value,
                        model.facts,
                        model.dense_read_bytes,
                        facts,
                        platform,
                        drive_key,
                        host_work_bytes=host_work_bytes,
                        gpu_work_bytes=gpu_work_bytes,
                        observed_hits=observed_hits,
                        graph_weights=weights if weights and weights.graph_store_bytes else None,
                    )
                    if model.facts.experts is not None
                    else dense_decode_estimate(
                        value,
                        model.dense_read_bytes,
                        facts,
                        model=model.facts,
                        weights=weights,
                        platform=platform,
                        drive_key=drive_key,
                        host_work_bytes=host_work_bytes,
                        gpu_work_bytes=gpu_work_bytes,
                    ),
                )
                for model in models
            ]
    return [(model, Estimate(None, reason)) for model in models]


def estimate_lines(rows: list[tuple[catalog.Model, Estimate]]) -> list[str]:
    return [
        f"estimate: {model.repo}:{model.quantization} "
        + (
            (
                f"{estimate.scenario_range[0]:.1f}–{estimate.scenario_range[1]:.1f} tok/s "
                "across timing scenarios; "
                if estimate.scenario_range
                else ""
            )
            + f"{estimate.reference_scenario.replace('_', ' ')} reference "
            + f"{estimate.tokens_per_second:.1f} tok/s ({estimate.detail})"
            if estimate.tokens_per_second is not None
            else f"none: {estimate.detail}"
        )
        for model, estimate in rows
    ]


def report(
    value: Characteristics,
    path: Path,
    drive_key: Optional[str],
    rows: list[tuple[catalog.Model, Estimate]],
) -> dict[str, object]:
    """The profile's summary and the estimates as one JSON object."""
    hard = value["identity"]["hard"]
    rates = [row["Bps"] for row in value["memory"]["dram_read_Bps"]]
    drive = next((row for row in value["storage"] if row["drive_key"] == drive_key), None)
    return {
        "profile": str(path),
        "written_utc": value["created_utc"],
        "machine": {
            "platform": hard["platform"],
            "cpu": hard["cpu_brand"],
            "logical_cpus": hard["logical_cpus"],
            "memory_bytes": hard["ram_bytes"],
            "memory_read_Bps": max(rates) if rates else None,
            "gpus": [gpu["name"] for gpu in hard["gpus"]],
            "models_drive": {
                "model": drive["model"],
                "read_Bps": max(row["Bps"] for row in drive["random_read"]),
            }
            if drive is not None
            else None,
        },
        "estimates": [
            {
                "model": model.repo,
                "quantization": model.quantization,
                "decode_tokens_per_second": estimate.tokens_per_second,
                "reference_scenario": estimate.reference_scenario
                if estimate.tokens_per_second is not None
                else None,
                "estimator_version": 2,
                "detail": estimate.detail,
                "assumptions": estimate.assumptions,
                "scenarios": estimate.scenarios,
                "components": estimate.components,
                "routing_scenarios": estimate.scenario_details,
                "scenario_range_tokens_per_second": estimate.scenario_range,
                "range_kind": "conditional scenarios; not a confidence interval",
            }
            for model, estimate in rows
        ],
    }


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
    runtime: Callable[[], RuntimeFacts] = read_runtime,
) -> int:
    if args.shards and args.estimate_gguf is None:
        raise ValueError("--shard/--ple requires --estimate-gguf")
    for name in ("memory_gib", "host_work_mib", "gpu_work_mib"):
        number = getattr(args, name)
        if number is not None and (not math.isfinite(number) or number < 0):
            raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative")
    original_runtime = runtime
    if args.memory_gib is not None:

        def budget_runtime() -> RuntimeFacts:
            return replace(original_runtime(), memory_budget_bytes=int(args.memory_gib * GIB))

        runtime = budget_runtime
    host = host or NativeHost()
    path = args.output or default_path(host.platform(), environ, home)
    paths = [str(model) for model in args.model]
    resources = args.resources
    if resources is None and args.build is None and (not args.show or args.estimate_gguf):
        resources = installed_resources(environ, home)
        if not resources.is_file():
            raise ValueError(
                "engine not installed; run scripts/install.sh "
                "or pass --build DIR or --resources FILE"
            )
    target = None
    weights = None
    observed_hits = None
    target_layers = 0
    if (args.calibrate_from or args.calibration) and not (
        args.estimate_gguf or args.estimate_facts
    ):
        raise ValueError("calibration requires --estimate-gguf or --estimate-facts")
    if args.routing_events and not (args.estimate_gguf or args.estimate_facts):
        raise ValueError("--routing-events requires --estimate-gguf or --estimate-facts")
    if args.estimate_gguf is not None or args.estimate_facts is not None:
        if args.estimate_gguf is not None:
            binary = (
                read_resources(resources).binary
                if resources is not None
                else str(args.build / "bin" / executable(host.platform(), "inferssd-server"))
            )
            raw = engine_model.query(
                binary,
                str(args.estimate_gguf),
                lambda argv: (runner or run_process)(argv, engine_model.TIMEOUT_S, environ),
                args.shards,
            )
            if not paths:
                paths = raw["files"]
        else:
            raw = engine_model.validate(json.loads(args.estimate_facts.read_text(encoding="utf-8")))
        if args.routing_events:
            layout = raw["experts"]
            if not layout or not layout["used"]:
                raise ValueError("--routing-events requires a model with known routed experts")
            with args.routing_events.open(encoding="utf-8") as stream:
                observed_hits = last_request_hits(
                    stream,
                    {layer for layer, _ in layout["layers"]},
                    layout["count"],
                    layout["used"],
                )
        target_layers = raw["layers"]
        facts = facts_from_engine(raw)
        weights = dense_weights(raw)
        read_bytes = weights.active_bytes
        # Report the actual GGUF types. Mixed checkpoints need not have a canonical quant label.
        formats = facts.experts.format_bytes if facts.experts else facts.layer_format_bytes
        target = catalog.Model(
            facts.paths[0],
            "+".join(name for name, _ in formats),
            "",
            facts,
            facts.dense_bytes - read_bytes,
        )

    def output(value: Characteristics, drive_key: Optional[str]) -> None:
        lines = summary(value, path)
        if args.json or args.model or target:
            sample = None
            estimate_runtime = lru_cache(maxsize=1)(runtime)
            rows = estimates(
                value,
                drive_key,
                host.platform(),
                estimate_runtime,
                host_work_bytes=int(args.host_work_mib * (1 << 20)),
                gpu_work_bytes=int(args.gpu_work_mib * (1 << 20)),
                target=target,
                weights=weights,
                observed_hits=observed_hits,
            )
            if args.calibrate_from or args.calibration:
                key = calibration.machine_key(value, estimate_runtime())
                if args.calibrate_from:
                    response = json.loads(args.calibrate_from.read_text(encoding="utf-8"))
                    sample = calibration.fit(response, rows[0][1], target_layers, key)
                else:
                    saved_calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
                    sample = calibration.load(saved_calibration, key)
                    rows = [(m, calibration.apply(e, sample, target_layers)) for m, e in rows]
            if args.json or args.calibrate_from:
                result = report(value, path, drive_key, rows)
                if sample is not None:
                    result["calibration"] = sample
                    result["calibration_applied"] = bool(args.calibration)
                emit(json.dumps(result, separators=(",", ":")))
                return
            lines += estimate_lines(rows)
        for line in lines:
            emit(line)

    if args.show:
        if (
            args.reprofile
            or args.estimate_compute
            or ((args.resources or args.build) and args.estimate_gguf is None)
        ):
            raise ValueError(
                "--show measures nothing; omit --reprofile, --resources, "
                "--build and --estimate-compute"
            )
        saved = load_current(path)
        if saved is None:
            raise ValueError(f"no machine profile at {path}; run inferssd profile first")
        output(saved, models_drive(model_drives(host, paths)) if paths else None)
        return 0
    suffix = ".exe" if host.platform() == "windows" else ""
    probes = Probes(
        tuple(locate(resources, args.build, suffix=suffix).items()), runner or run_process
    )
    with exclusive(path):
        try:
            previous = load_current(path)
        except (ContractError, UnicodeDecodeError) as error:
            progress(f"ignoring the unreadable saved profile {path}: {error}")
            previous = None
        progress("reading machine identity")
        surveyed = survey(host, paths, previous, args.reprofile, progress)
        value = profile(
            host,
            probes,
            paths,
            previous,
            args.reprofile,
            clock=clock,
            progress=progress,
            surveyed=surveyed,
        )
        if args.estimate_compute:
            value = measure_decode_compute(value, probes, runtime(), progress, args.reprofile)
        if value is not previous:
            save(value, path)
    output(value, models_drive(surveyed.drives))
    changed = _changed(value, previous)
    partial = [name for name, rec in changed.items() if rec["status"] == "partial"]
    failed = [name for name, rec in changed.items() if rec["status"] == "failed"]
    # With --json, stdout holds the one JSON object; the run's problems go to stderr.
    note = progress if args.json or args.calibrate_from else emit
    if partial:
        note(f"{len(partial)} section(s) partial: {', '.join(partial)}")
    if failed:
        note(f"{len(failed)} section(s) failed: {', '.join(failed)}")
    return 1 if failed else 0
