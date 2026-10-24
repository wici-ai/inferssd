#!/usr/bin/env python3
"""Run one explicit preparation command under the existing device ownership runtime.

This helper is for independent release comparisons, not a CI device or runner.
The JSON specification and its output directory are supplied by the operator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import time

from inferssd.engine.owned import OwnedProcess
from inferssd.launch.types import Resolution
from inferssd.platform.policy import Policy
from inferssd.platform.run_directory import reserve
from inferssd.platform.worker import cleanup_confirmed, save


def run(specification: dict, output: Path) -> int:
    argv = specification["argv"]
    if not isinstance(argv, list) or not argv or any(not isinstance(x, str) or not x for x in argv):
        raise ValueError("argv must be an explicit nonempty argument list")
    policy = Policy.read(specification["policy"])
    if policy.platform != "linux":
        raise ValueError("this release preparation helper currently supports Linux")
    environment = dict(specification["environment"])
    resolution = Resolution(tuple(argv), tuple(environment.items()), tuple(environment.items()),
                            tuple(specification["scope"]), (), (), None, 0)
    reservation = reserve(str(output))
    save(output / "command.json", specification)
    process = None
    signals = []
    handlers = {number: signal.signal(number, lambda signum, _: signals.append(signum))
                for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    result = {"status": "failed", "purpose": "release-comparison-preparation", "model_requests": 0}
    deadline = time.monotonic() + policy.runtime_seconds
    try:
        process = OwnedProcess(resolution, str(output), policy, reservation=reservation,
                               command_mode=True)
        while process.poll() is None:
            if signals:
                raise InterruptedError("preparation controller interrupted")
            if time.monotonic() >= deadline:
                raise TimeoutError("preparation command deadline exceeded")
            time.sleep(.5)
        if process.poll() != 0:
            raise RuntimeError("preparation command failed; inspect server.log and ownership.json")
        result["status"] = "passed"
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if process is not None:
            try:
                process.stop()
                result["cleanup_confirmed"] = True
            except BaseException as error:
                result.update(status="failed", cleanup_confirmed=cleanup_confirmed(output / "ownership.json"),
                              cleanup_error=f"{type(error).__name__}: {error}")
        save(output / "command-result.json", result)
        for number, handler in handlers.items():
            signal.signal(number, handler)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    raise SystemExit(run(json.loads(options.spec.read_text()), options.output.resolve()))
