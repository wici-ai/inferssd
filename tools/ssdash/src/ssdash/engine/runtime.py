"""Explicit platform launch adapter and Linux Controller factory.

The factory is opt-in; old production and online entry points remain unchanged.
Darwin currently executes static M1 launches. Mac adaptive placement (M2) is not
implemented by applying Linux Machine.env variables to a Mac server.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import BinaryIO, TextIO, cast

from ssdash.engine.core import Engine, NativeEffects, post
from ssdash.engine.owned import OwnedProcess
from ssdash.launch.darwin import resolve_darwin
from ssdash.launch.linux import resolve_linux
from ssdash.launch.live import Host, LiveArtifacts
from ssdash.launch.prepare import prepare
from ssdash.launch.recorded import run_indexer
from ssdash.launch.types import LaunchInput, Output, Resolution
from ssdash.loop.controller import Controller
from ssdash.loop.records import ObservationSession
from ssdash.model.machine import Machine
from ssdash.plan.recorded import DirectoryArtifacts
from ssdash.platform.policy import Policy
from ssdash.telemetry.files import LocalArmFiles


class LocalEffects:
    def __init__(self, environment: Mapping[str, str], stdout: TextIO) -> None:
        self.environment, self.stdout = environment, stdout

    def open(self, path: str, mode: str = "r") -> TextIO:
        return cast(TextIO, open(path, mode))

    def makedirs(self, path: str, *, exist_ok: bool = False) -> None:
        os.makedirs(path, exist_ok=exist_ok)

    def getenv(self, name: str, default: str) -> str:
        return self.environment.get(name, default)

    def clock(self) -> float:
        return time.time()


class LocalObservations(LocalArmFiles):
    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def stat(self, path: str) -> os.stat_result:
        return os.stat(path)

    def open_binary(self, path: str) -> BinaryIO:
        return open(path, "rb")


class Runtime:
    def __init__(
        self, inputs: LaunchInput, policy: Policy, *, host: Host, emit: Callable[[Output], None]
    ) -> None:
        self.inputs, self.policy, self.host, self.emit = inputs, policy, host, emit

    def resolve(self, environment: Mapping[str, str], args: Sequence[str]) -> Resolution:
        inputs = LaunchInput(
            self.inputs.repo,
            tuple(environment.items()),
            tuple(args),
            self.inputs.profile,
            self.inputs.mode,
        )
        adapter = LiveArtifacts(
            inputs, host=self.host, run=run_indexer, replace=os.replace, python=sys.executable
        )
        value = prepare(
            inputs,
            resolver=(
                partial(resolve_linux, execute=True)
                if self.policy.platform == "linux"
                else resolve_darwin
            ),
            observe=adapter,
            emit=self.emit,
            darwin_errors=self.policy.platform == "darwin",
        )
        if value.code != 0:
            raise RuntimeError("native launch preparation failed")
        return value

    def engine(
        self, directory: str, port: int, tier: int, cpus: str, envs: str, args: list[str]
    ) -> Engine:
        def checked(environment: Mapping[str, str], extra: Sequence[str]) -> Resolution:
            value = self.resolve(environment, extra)
            ports = [
                value.argv[i + 1] for i, item in enumerate(value.argv[:-1]) if item == "--port"
            ]
            hosts = [
                value.argv[i + 1] for i, item in enumerate(value.argv[:-1]) if item == "--host"
            ]
            if not ports or ports[-1] != str(port) or not hosts or hosts[-1] != "127.0.0.1":
                raise ValueError("owned Engine requires its exact loopback health host and port")
            return value

        effects = NativeEffects()

        def own(result: Resolution, path: str) -> OwnedProcess:
            assert effects.reservation is not None
            return OwnedProcess(result, path, self.policy, reservation=effects.reservation)

        return Engine(
            directory,
            port,
            tier,
            cpus,
            envs,
            args,
            environment=dict(self.inputs.environment),
            effects=effects,
            resolve=checked,
            own=own,
        )

    def controller(
        self,
        run: str,
        out: str,
        port: int,
        start: tuple[int, str],
        args: Sequence[str],
        *,
        host_threads: Callable[[], int],
        stdout: TextIO,
        loop: bool = True,
        window: tuple[int, int] = (129, 512),
        decode: int = 512,
        horizon: int = 3,
    ) -> Controller:
        if self.policy.platform != "linux":
            raise ValueError(
                "Darwin adaptive Controller requires the unimplemented M2 placement adapter"
            )
        return Controller(
            run,
            out,
            port,
            start,
            args,
            loop,
            window,
            decode,
            horizon,
            machine_factory=lambda path: Machine(
                DirectoryArtifacts(Path(path)), host_threads=host_threads
            ),
            engine_factory=self.engine,
            post=post,
            effects=LocalEffects(dict(self.inputs.environment), stdout),
            session=ObservationSession(files=LocalObservations()),
        )
