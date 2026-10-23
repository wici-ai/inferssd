"""Formal launch dry-run command. Native execution has no default or fallback."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from inferssd.launch.darwin import resolve_darwin
from inferssd.launch.linux import resolve_linux
from inferssd.launch.prepare import prepare
from inferssd.launch.recorded import RecordedArtifacts, run_indexer
from inferssd.launch.types import LaunchInput, Output


def add_command(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    launch = commands.add_parser(
        "launch-recorded", help="historical F resolver comparison; never execute a server"
    )
    execution = launch.add_mutually_exclusive_group(required=True)
    execution.add_argument("--dry-run", action="store_true")
    launch.set_defaults(execute=False)
    launch.add_argument("--platform", choices=("linux", "darwin"), required=True)
    launch.add_argument("--repo", type=Path, required=True)
    launch.add_argument("--host-facts", type=Path)
    launch.add_argument("--runtime-policy", type=Path)
    launch.add_argument("--output-dir", type=Path)
    launch.add_argument("--port", type=int)
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


def launch_inputs(args: argparse.Namespace) -> LaunchInput:
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
    if args.port is not None:
        if not 1 <= args.port <= 65535:
            raise ValueError("--port must be in 1..65535")
        environment["LLAMA_SERVER_PORT" if args.platform == "linux" else "SSD_LLAMA_PORT"] = str(
            args.port
        )
    return LaunchInput(
        str(args.repo.resolve()), tuple(environment.items()), tuple(extra), args.profile, args.mode
    )


def run_launch(args: argparse.Namespace) -> int:
    if args.execute:
        raise ValueError("historical launch resolver cannot execute a server")
    if args.host_facts is None:
        raise ValueError("--dry-run requires --host-facts; it never observes the live host")
    inputs = launch_inputs(args)
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
