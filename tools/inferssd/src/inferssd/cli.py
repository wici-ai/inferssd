"""Opt-in recorded framework and source artifact commands."""

from __future__ import annotations

import argparse
import glob
import gzip
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from inferssd.bootstrap.core_sets import run_coresets
from inferssd.bootstrap.tier_slots import run_ubprobe
from inferssd.configuration.command import add_command as add_configuration_command
from inferssd.configuration.command import run as run_configuration
from inferssd.contracts import DeviceProfile
from inferssd.launch.command import add_command as add_historical_launch_command
from inferssd.launch.command import run_launch
from inferssd.launch.typed_command import add_command as add_launch_command
from inferssd.launch.typed_command import run as run_typed_launch
from inferssd.loop.report import run_report
from inferssd.model.machine import Machine
from inferssd.plan.planner import run_plan
from inferssd.plan.planner_types import PlanArguments
from inferssd.plan.recorded import DirectoryArtifacts, explicit_host_threads, write_json
from inferssd.plan.settings_command import SettingsArguments, run_settings
from inferssd.plan.settings_recorded import write_settings
from inferssd.probe.calibrate import CalibrationArguments, run_calibration
from inferssd.probe.device_profile import aggregate
from inferssd.probe.recorded import load_reports
from inferssd.source.command import add_commands as add_source_commands
from inferssd.source.command import run_source
from inferssd.telemetry.files import LOCAL_FILES, lines
from inferssd.telemetry.recorded import ledger, read_arm


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="InferSSD recorded tools and explicit owned launch")
    commands = parser.add_subparsers(dest="command", required=True)
    add_source_commands(commands)
    add_launch_command(commands)
    add_historical_launch_command(commands)
    add_configuration_command(commands)
    bootstrap = commands.add_parser(
        "bootstrap-records", help="read recorded CPU rows and ubatch logs; never run a probe"
    )
    bootstrap_commands = bootstrap.add_subparsers(dest="bootstrap_command", required=True)
    coresets = bootstrap_commands.add_parser("coresets", help="CPU candidates from recorded labels")
    coresets.add_argument("cpu_rows")
    slots = bootstrap_commands.add_parser("tier-slots", help="tier slots from recorded server logs")
    slots.add_argument("pipeline")
    slots.add_argument("ubs", nargs="*")
    report = commands.add_parser("report", help="compare recorded conversations only")
    report.add_argument("run")
    report.add_argument(
        "--host-threads", type=int, help="explicit fallback for a recorded split placement"
    )
    probe = commands.add_parser("probe", help="aggregate existing report files only")
    probe.add_argument("--from-reports", required=True, type=Path)
    probe.add_argument("--experts", required=True, type=Path)
    validate = commands.add_parser("validate", help="validate a device schema-2 artifact")
    validate.add_argument("--device", required=True, type=Path)
    plan = commands.add_parser("plan", help="plan from existing bootstrap artifacts only")
    plan.add_argument("run", type=Path)
    plan.add_argument("--prompts", required=True)
    plan.add_argument("--decode", type=int, default=512)
    plan.add_argument("--window", default="129-512")
    plan.add_argument("--known", action="store_true")
    plan.add_argument("--pending", default="")
    plan.add_argument(
        "--host-threads", type=int, help="explicit fallback for a recorded split placement"
    )
    plan.add_argument("-o", "--out", required=True)
    settings = commands.add_parser(
        "settings", help="derive legacy settings.env from recorded artifacts only"
    )
    settings.add_argument("run")
    settings.add_argument("--tier", type=int)
    settings.add_argument("--cpus")
    settings.add_argument("--prompts")
    settings.add_argument("--decode", type=int)
    settings.add_argument("--ledgers", nargs="*", default=[])
    settings.add_argument("-o", "--out")
    settings.add_argument(
        "--host-threads", type=int, help="explicit fallback for a recorded split placement"
    )
    fit = commands.add_parser("fit", help="calibrate from recorded run logs only (no bootstrap)")
    fit.add_argument("--run", action="append", required=True, help="NAME=DIR; first is reference")
    fit.add_argument("--skip", type=int, default=128)
    fit.add_argument("--graph")
    fit.add_argument("--ubatch-probe")
    fit.add_argument("--profile", required=True)
    fit.add_argument("-o", "--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "launch":
        try:
            return run_typed_launch(args)
        except (OSError, ValueError, RuntimeError) as error:
            parser.error(str(error))
    if args.command == "launch-recorded":
        try:
            return run_launch(args)
        except (OSError, ValueError, RuntimeError) as error:
            parser.error(str(error))
    if args.command == "configuration":
        try:
            return run_configuration(args)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    if args.command == "bootstrap-records":
        if args.bootstrap_command == "coresets":
            run_coresets(["coresets.py", args.cpu_rows], open_file=open, emit=print)
        else:
            run_ubprobe(
                ["ubprobe.py", args.pipeline, *args.ubs],
                open_file=open,
                open_gzip=gzip.open,
                emit=print,
            )
        return 0
    if args.command == "report":
        run_report(
            args.run,
            machine_factory=lambda run: Machine(
                DirectoryArtifacts(Path(run)),
                host_threads=explicit_host_threads(args.host_threads),
            ),
            open_file=open,
            glob_paths=glob.glob,
            exists=os.path.exists,
            basename=os.path.basename,
            read_lines=lambda path: lines(path, LOCAL_FILES),
            emit=print,
        )
        return 0
    if args.command == "source":
        run_source(args)
        return 0
    if args.command == "fit":
        run_calibration(
            CalibrationArguments(
                args.run, args.skip, args.profile, args.out, args.graph, args.ubatch_probe
            ),
            read_arm=read_arm,
            open_file=open,
            emit=print,
        )
        return 0
    if args.command == "settings":
        run_settings(
            SettingsArguments(
                args.run, args.tier, args.cpus, args.prompts, args.decode, args.ledgers, args.out
            ),
            DirectoryArtifacts(Path(args.run)),
            host_threads=explicit_host_threads(args.host_threads),
            ledger=ledger,
            absolute_path=os.path.abspath,
            write_text=write_settings,
            stdout=sys.stdout.write,
        )
        return 0
    if args.command == "plan":
        run_plan(
            PlanArguments(
                args.prompts, args.decode, args.window, args.known, args.pending, args.out
            ),
            DirectoryArtifacts(args.run),
            host_threads=explicit_host_threads(args.host_threads),
            clock=time.time,
            emit=print,
            write_json=write_json,
        )
        return 0
    try:
        if args.command == "probe":
            reports, experts = load_reports(args.from_reports, args.experts)
            print(aggregate(reports, experts).to_json(), end="")
        else:
            DeviceProfile.from_json(args.device.read_text())
            print("device schema 2: valid")
    except (OSError, ValueError, KeyError, IndexError) as error:
        parser.error(str(error))
    return 0
