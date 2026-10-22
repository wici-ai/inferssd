"""Offline entry points for the first framework extraction batch."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from ssdash.contracts import DeviceProfile
from ssdash.probe.device_profile import aggregate
from ssdash.probe.recorded import load_reports


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="SSDash recorded-device tools (no hardware probes)"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("probe", help="aggregate existing report files only")
    probe.add_argument("--from-reports", required=True, type=Path)
    probe.add_argument("--experts", required=True, type=Path)
    validate = commands.add_parser("validate", help="validate a device schema-2 artifact")
    validate.add_argument("--device", required=True, type=Path)
    args = parser.parse_args(argv)
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
