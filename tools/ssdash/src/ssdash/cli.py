"""Opt-in recorded framework and source artifact commands."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from ssdash.contracts import DeviceProfile
from ssdash.plan.planner import run_plan
from ssdash.plan.planner_types import PlanArguments
from ssdash.plan.recorded import DirectoryArtifacts, explicit_host_threads, write_json
from ssdash.plan.settings_command import SettingsArguments, run_settings
from ssdash.plan.settings_recorded import write_settings
from ssdash.probe.calibrate import CalibrationArguments, run_calibration
from ssdash.probe.device_profile import aggregate
from ssdash.probe.recorded import load_reports
from ssdash.source.command import add_commands as add_source_commands
from ssdash.source.command import run_source
from ssdash.telemetry.recorded import ledger, read_arm


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="SSDash recorded-device tools (no hardware probes)"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    add_source_commands(commands)
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
