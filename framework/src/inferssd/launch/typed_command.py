"""Explicit typed production launch and recorded-host preparation commands."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import asdict
from functools import partial
from pathlib import Path

from inferssd.launch.prepare import prepare
from inferssd.launch.recorded import RecordedArtifacts, run_indexer
from inferssd.launch.resources import read_resources
from inferssd.launch.typed import LaunchSpec, resolve_typed
from inferssd.launch.types import LaunchInput, Output
from inferssd.platform.policy import Policy
from inferssd.platform.run_directory import reserve


def add_command(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    launch = commands.add_parser(
        "launch", help="launch from typed settings with explicit process ownership"
    )
    mode = launch.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    launch.add_argument(
        "--platform",
        choices=("linux", "darwin"),
        default="darwin" if sys.platform == "darwin" else "linux",
        help="defaults to this host",
    )
    launch.add_argument("--repo", type=Path, help="optional development checkout adapter")
    launch.add_argument("--resources", type=Path, help="inferssd-engine 1 deployment descriptor")
    launch.add_argument("--template", type=Path, help="explicit chat template, overrides resources")
    launch.add_argument("--model", type=Path, required=True)
    launch.add_argument("--runtime-policy", type=Path, required=True)
    launch.add_argument(
        "--output-dir",
        type=Path,
        help="new directory for this launch's logs and receipts (default: runs/<UTC time>)",
    )
    launch.add_argument(
        "--host-facts", type=Path, help="required recorded host observations for dry-run"
    )
    launch.add_argument("--settings", type=Path, help="inferssd-settings 1 input file")
    launch.add_argument(
        "--set", dest="overrides", action="append", default=[], help="typed key=literal override"
    )
    launch.add_argument("--binary", type=Path)
    launch.add_argument("--artifact-dir", type=Path)
    launch.add_argument("--profile", choices=("dsv", "muse"), default="muse")
    launch.add_argument("--port", type=int, default=8080)
    launch.add_argument("--context", type=int, default=8192)
    launch.add_argument("--batch", type=int, default=512)
    launch.add_argument("--ubatch", type=int, default=512)
    launch.add_argument(
        "--batch-threads", type=int, help="native prefill threads (1..256; defaults to cpu.threads)"
    )
    launch.add_argument("--result", type=Path, help="dry-run resolution receipt")
    launch.add_argument("extra", nargs=argparse.REMAINDER)
    launch.set_defaults(mode="server")


def specification(args: argparse.Namespace) -> LaunchSpec:
    overrides = []
    for item in args.overrides:
        key, equal, value = item.partition("=")
        if not key or not equal:
            raise ValueError("--set requires key=literal")
        overrides.append((key, value))
    return LaunchSpec(
        str(args.model.resolve()),
        args.port,
        str(args.settings.resolve()) if args.settings else None,
        tuple(overrides),
        str(args.binary.resolve()) if args.binary else None,
        str(args.artifact_dir.resolve()) if args.artifact_dir else None,
        args.context,
        args.batch,
        args.ubatch,
        args.batch_threads,
        str(args.template.resolve()) if args.template else None,
        read_resources(args.resources) if args.resources else None,
    )


def inputs(args: argparse.Namespace) -> LaunchInput:
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    return LaunchInput(
        str(args.repo.resolve()) if args.repo else "",
        tuple(os.environ.items()),
        tuple(extra),
        args.profile,
    )


def default_output_directory() -> Path:
    """A new directory under ./runs; an existing run directory is never reused."""
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return Path("runs") / (stamp + "-" + uuid.uuid4().hex[:6])


def run(args: argparse.Namespace) -> int:
    launch_input = inputs(args)
    if args.output_dir is None:
        args.output_dir = default_output_directory()
        print("run directory: " + str(args.output_dir.resolve()), file=sys.stderr)
    if args.execute:
        from inferssd.engine.command import run_native

        return run_native(args, launch_input)
    if args.host_facts is None:
        raise ValueError(
            "--dry-run requires recorded --host-facts and never observes the live host"
        )
    host = json.loads(args.host_facts.read_text(encoding="utf-8"))
    if not isinstance(host, dict):
        raise ValueError("recorded host facts must be an object")
    data = json.loads(args.runtime_policy.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("runtime policy must be an object")
    policy = Policy.read(data)
    if policy.platform != args.platform:
        raise ValueError("launch platform differs from policy")
    claim = reserve(str(args.output_dir))
    adapter = RecordedArtifacts(
        launch_input, host, run=run_indexer, replace=os.replace, python=sys.executable
    )

    def emit(event: Output) -> None:
        (sys.stdout if event.channel == "stdout" else sys.stderr).write(event.text)

    result = prepare(
        launch_input,
        resolver=partial(
            resolve_typed,
            spec=specification(args),
            platform=args.platform,
            directory=str(claim.directory),
            memory=policy.memory_budget_bytes,
        ),
        observe=adapter,
        emit=emit,
    )
    claim.verify()
    if args.result:
        # Environment values stay private to the server process and never enter
        # the dry-run receipt. Only explicitly chosen settings/argv are written.
        receipt = asdict(result)
        receipt["effective_environment"] = [name for name, _ in result.effective_environment]
        with args.result.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    return int(result.code or 0)
