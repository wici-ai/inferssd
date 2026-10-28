"""Explicit schema/check/write commands for the production settings v1 format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from inferssd.configuration.settings import PLATFORMS, parse, render, schema


def add_command(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    config = commands.add_parser("configuration", help="typed SSD settings v1 schema/check/write")
    config.add_argument("action", choices=("schema", "check", "write"))
    config.add_argument(
        "--platform",
        choices=PLATFORMS,
        default={"darwin": "darwin", "win32": "windows"}.get(sys.platform, "linux"),
        help="defaults to this host",
    )
    config.add_argument("--input", type=Path)
    config.add_argument("--set", dest="overrides", action="append", default=[])
    config.add_argument("--output", type=Path)


def run(args: argparse.Namespace) -> int:
    if args.action == "schema":
        print(json.dumps(schema(), ensure_ascii=False, indent=2))
        return 0
    overrides = []
    for item in args.overrides:
        key, equal, literal = item.partition("=")
        if not equal:
            raise ValueError("--set requires key=literal")
        overrides.append((key, literal))
    value = parse(
        args.input.read_text() if args.input else None,
        str(args.input) if args.input else "",
        overrides=overrides,
        platform=args.platform,
    )
    if args.action == "check":
        print(json.dumps(value.document(), ensure_ascii=False, indent=2))
    else:
        if args.output is None:
            raise ValueError("configuration write requires --output")
        with args.output.open("x") as stream:
            stream.write(render(value))
    return 0
