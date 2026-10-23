"""Opt-in native launch command; dry-run never imports or invokes this path."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from typing import Optional

from ssdash.engine.core import Engine
from ssdash.engine.runtime import Runtime
from ssdash.launch.live import NativeHost
from ssdash.launch.types import LaunchInput, Output
from ssdash.platform.policy import Policy


def run_native(args: argparse.Namespace, inputs: LaunchInput) -> int:
    if args.result is not None:
        raise ValueError("--result is dry-run only; execution writes receipts in --output-dir")
    if args.host_facts is not None:
        raise ValueError("--execute observes the live host; recorded --host-facts are dry-run only")
    if args.runtime_policy is None or args.output_dir is None or args.port is None:
        raise ValueError("--execute requires --runtime-policy, --output-dir and --port")
    if inputs.mode != "server":
        raise ValueError("owned execution currently supports server mode only")
    data = json.loads(args.runtime_policy.read_text())
    if not isinstance(data, dict):
        raise ValueError("runtime policy must be a JSON object")
    policy = Policy.read(data)
    if policy.platform != args.platform:
        raise ValueError("--platform differs from runtime policy")

    def emit(event: Output) -> None:
        stream = sys.stdout if event.channel == "stdout" else sys.stderr
        stream.write(event.text)
        stream.flush()

    def interrupted(number: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"signal {number}")

    runtime = Runtime(inputs, policy, host=NativeHost(), emit=emit)
    engine: Optional[Engine] = None
    handlers = {
        number: signal.signal(number, interrupted)
        for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        engine = runtime.engine(
            str(args.output_dir), args.port, 0, "static", "", list(inputs.extra)
        )
        if not engine.ok:
            raise RuntimeError("owned server did not become healthy")
        while engine.alive():
            time.sleep(0.5)
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        try:
            if engine is not None:
                engine.stop()
        finally:
            for number, previous in handlers.items():
                signal.signal(number, previous)
