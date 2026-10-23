"""Formal launch dry-run command. Native execution has no default or fallback."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from ssdash.launch.darwin import resolve_darwin
from ssdash.launch.linux import resolve_linux
from ssdash.launch.prepare import prepare
from ssdash.launch.recorded import RecordedArtifacts, run_indexer
from ssdash.launch.types import LaunchInput, Output


def add_command(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    launch = commands.add_parser(
        "launch", help="resolve launch arguments and indexes without starting a server"
    )
    launch.add_argument("--dry-run", action="store_true", required=True)
    launch.add_argument("--platform", choices=("linux", "darwin"), required=True)
    launch.add_argument("--repo", type=Path, required=True)
    launch.add_argument("--host-facts", type=Path, required=True)
    launch.add_argument("--model-conf", type=Path)
    launch.add_argument("--settings", type=Path)
    launch.add_argument("--model", type=Path)
    launch.add_argument("--model-dir", type=Path)
    launch.add_argument("--artifact-dir", type=Path)
    launch.add_argument("--profile", choices=("dsv", "muse"), default="muse")
    launch.add_argument("--mode", choices=("chat", "server"), default="server")
    launch.add_argument(
        "--result", type=Path, help="write ordered argv/environment/scope and observations as JSON"
    )
    launch.add_argument("extra", nargs=argparse.REMAINDER)


def run_launch(args: argparse.Namespace) -> int:
    environment = dict(os.environ)
    for name, value in (
        ("MODEL_CONF", args.model_conf),
        ("SETTINGS", args.settings),
        ("MODEL_DIR", args.model_dir),
    ):
        if value is not None:
            environment[name] = str(value)
    if args.model is not None:
        environment["MODEL" if args.platform == "linux" else "SSD_LLAMA_MODEL"] = str(args.model)
    if args.artifact_dir is not None:
        environment["EXPERT_INDEX" if args.platform == "linux" else "XDG_CACHE_HOME"] = str(
            args.artifact_dir
        )
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    inputs = LaunchInput(
        str(args.repo.resolve()), tuple(environment.items()), tuple(extra), args.profile, args.mode
    )
    host = json.loads(args.host_facts.read_text())
    if not isinstance(host, dict):
        raise ValueError("Host facts must be a JSON object")
    adapter = RecordedArtifacts(
        inputs, host, run=run_indexer, replace=os.replace, python=sys.executable
    )

    def emit(event: Output) -> None:
        stream = sys.stdout if event.channel == "stdout" else sys.stderr
        stream.write(event.text)
        stream.flush()

    result = prepare(
        inputs,
        resolver=resolve_linux if args.platform == "linux" else resolve_darwin,
        observe=adapter,
        emit=emit,
        darwin_errors=args.platform == "darwin",
    )
    if args.result is not None:
        args.result.write_text(json.dumps(asdict(result), indent=2) + "\n")
    assert result.code is not None
    return result.code
