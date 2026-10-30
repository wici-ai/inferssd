# SPDX-License-Identifier: Apache-2.0
"""Friendly commands that prepare inputs for the existing owned launcher."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

from inferssd.app.hub import download, selection
from inferssd.app.memory import admit, aligned_budget, available_memory
from inferssd.launch import typed_command
from inferssd.launch.resources import read_resources
from inferssd.platform.policy import Policy

GIB = 1024**3


def data_directory() -> Path:
    return Path(os.environ.get("INFERSSD_HOME", Path.home() / ".local/share/inferssd"))


def add_commands(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    for mode in ("cli", "serve"):
        parser = commands.add_parser(
            mode,
            help="download a tested GGUF and "
            + ("chat in the terminal" if mode == "cli" else "start the OpenAI-compatible API"),
        )
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument(
            "-hf", "--hf-repo", help="README Hugging Face repository[:quantization]"
        )
        source.add_argument("-m", "--model", type=Path, help="local first GGUF shard (no download)")
        parser.add_argument("--revision", default="main", help="Hugging Face revision")
        parser.add_argument("--model-dir", type=Path, default=data_directory() / "models")
        parser.add_argument("--resources", type=Path, help="engine descriptor (installer sets it)")
        parser.add_argument(
            "--runtime-policy", type=Path, help="override the automatic memory policy"
        )
        parser.add_argument("--memory-gib", type=float, help="server RAM budget in GiB")
        parser.add_argument("--runtime-seconds", type=int, default=86400, help="default: 24 hours")
        parser.add_argument("--port", type=int, default=8080)
        parser.add_argument("--context", type=int, default=8192)
        parser.add_argument("--settings", type=Path)
        parser.add_argument("--set", dest="overrides", action="append", default=[])
        parser.add_argument("--machine", type=Path)
        parser.add_argument("--reprofile", action="store_true")
        if mode == "cli":
            parser.add_argument("-p", "--prompt", help="answer one prompt and exit")
            parser.add_argument("-n", "--max-tokens", type=int, default=512)


def physical_memory() -> int:
    if sys.platform == "darwin":
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    raise ValueError("cannot read system RAM; provide --runtime-policy")


def default_policy(total: int, budget: int, state: Path, seconds: int) -> Policy:
    reserve = max(2 * GIB, total // 5)
    budget = aligned_budget(budget)
    if budget <= 0 or budget > total - reserve:
        raise ValueError("server RAM budget must leave at least 2 GiB / 20% for the host")
    return Policy.read(
        {
            "platform": sys.platform,
            "lock_path": str(state / "runtime.lock"),
            "stop_path": str(state / "runtime.stop"),
            "memory_budget_bytes": budget,
            "host_reserve_bytes": reserve,
            "minimum_memlock_bytes": 0,
            "max_compressor_growth_bytes": 0,
            "max_swap_growth_bytes": 0,
            "runtime_seconds": seconds,
            "cleanup_seconds": 30,
        }
    )


def run(args: argparse.Namespace) -> int:
    from inferssd.app.chat import chat
    from inferssd.engine.command import run_native

    if sys.platform not in ("linux", "darwin"):
        raise ValueError("cli and serve currently support Linux CUDA and Apple Silicon macOS")
    if not 1 <= args.port <= 65535 or args.context < 1 or args.runtime_seconds < 1:
        raise ValueError("port must be 1..65535; context and runtime-seconds must be positive")
    if args.command == "cli" and args.max_tokens < 1:
        raise ValueError("max-tokens must be positive")
    if args.machine and args.reprofile:
        raise ValueError("choose --machine or --reprofile")
    if args.runtime_policy and args.memory_gib is not None:
        raise ValueError("choose --runtime-policy or --memory-gib")
    if args.memory_gib is not None and not math.isfinite(args.memory_gib):
        raise ValueError("memory-gib must be a finite positive number")
    resources = args.resources or Path(
        os.environ.get("INFERSSD_RESOURCES", data_directory() / "current/engine-resources.json")
    )
    if not resources.is_file():
        raise ValueError("engine not installed; run scripts/install.sh or pass --resources FILE")
    deployment = read_resources(resources)
    if deployment.platform is not None and deployment.platform != sys.platform:
        raise ValueError("engine resources target a different platform")
    if not Path(deployment.binary).is_file() or not os.access(deployment.binary, os.X_OK):
        raise ValueError("engine server is missing or not executable; rerun the installer")
    for _, template in deployment.templates:
        if not Path(template).is_file():
            raise ValueError("engine chat template is missing; rerun the installer")
    state = (data_directory() / "state").resolve()
    state.mkdir(parents=True, exist_ok=True)
    if args.runtime_policy:
        policy_path = args.runtime_policy.resolve()
        policy = Policy.read(json.loads(policy_path.read_text()))
        if policy.platform != sys.platform:
            raise ValueError("runtime policy platform differs from this host")
        if aligned_budget(policy.memory_budget_bytes) != policy.memory_budget_bytes:
            raise ValueError("runtime policy memory_budget_bytes must be page-aligned on Linux")
    else:
        total = physical_memory()
        requested = int(args.memory_gib * GIB) if args.memory_gib is not None else None
        admission = admit(total, available_memory(), requested)
        policy = default_policy(total, admission.memory_budget_bytes, state, args.runtime_seconds)
        policy_path = state / ("policy-" + uuid.uuid4().hex + ".json")
        # Keep the policy alongside launch receipts for reproducibility.
        policy_path.write_text(json.dumps(asdict(policy), indent=2) + "\n")
        policy_path.with_suffix(".memory.json").write_text(
            json.dumps(asdict(admission), indent=2) + "\n"
        )
        print(
            f"Available RAM: {admission.available_bytes / GIB:.2f} GiB; "
            f"startup margin: {admission.startup_margin_bytes / GIB:.2f} GiB "
            "(in addition to host reserve).",
            file=sys.stderr,
        )
    print(
        f"Server RAM budget: {policy.memory_budget_bytes / GIB:.1f} GiB; "
        f"host reserve: {policy.host_reserve_bytes / GIB:.1f} GiB; "
        f"time limit: {policy.runtime_seconds}s. Override with --runtime-policy.",
        file=sys.stderr,
    )
    if args.hf_repo:
        selection(args.hf_repo)
        model = download(args.hf_repo, args.revision, args.model_dir.expanduser().resolve())
    else:
        model = args.model.expanduser().resolve(strict=True)
    parser = argparse.ArgumentParser()
    typed_command.add_command(parser.add_subparsers(dest="command", required=True))
    command = [
        "launch",
        "--execute",
        "--resources",
        str(resources.resolve()),
        "--runtime-policy",
        str(policy_path),
        "--model",
        str(model),
        "--port",
        str(args.port),
        "--context",
        str(args.context),
        "--output-dir",
        str(state / "runs" / uuid.uuid4().hex),
    ]
    for name in ("settings", "machine"):
        if getattr(args, name):
            command.extend(["--" + name, str(getattr(args, name))])
    for value in args.overrides:
        command.extend(["--set", value])
    if args.reprofile:
        command.append("--reprofile")
    launch = parser.parse_args(command)
    if args.command == "serve":
        print(f"Starting API at http://127.0.0.1:{args.port}/v1", file=sys.stderr)
        return typed_command.run(launch)
    return run_native(
        launch,
        typed_command.inputs(launch),
        session=lambda: chat(args.port, args.prompt, args.max_tokens),
    )
