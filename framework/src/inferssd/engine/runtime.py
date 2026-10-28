"""Typed production launch and Linux controller factories.

Darwin executes static M1 launches. Online adaptive Mac M2 remains deferred.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import BinaryIO, Optional, TextIO, cast

from inferssd.configuration.derive import Selection
from inferssd.configuration.settings import Snapshot, parse
from inferssd.engine.core import Engine, NativeEffects, post
from inferssd.engine.owned import OwnedProcess
from inferssd.launch.live import Host, LiveArtifacts
from inferssd.launch.prepare import prepare
from inferssd.launch.recorded import run_indexer
from inferssd.launch.typed import LaunchSpec, reject_legacy_environment, resolve_typed
from inferssd.launch.types import LaunchInput, Output, Resolution
from inferssd.loop.controller import Controller
from inferssd.loop.typed_controller import TypedController
from inferssd.model.typed import TypedMachine
from inferssd.plan.recorded import DirectoryArtifacts
from inferssd.platform.policy import Policy
from inferssd.telemetry.controller import NativeObservationSession
from inferssd.telemetry.files import LocalArmFiles


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
        self,
        inputs: LaunchInput,
        policy: Policy,
        *,
        host: Host,
        emit: Callable[[Output], None],
        configuration: Optional[LaunchSpec] = None,
    ) -> None:
        self.inputs, self.policy, self.host, self.emit = inputs, policy, host, emit
        self.configuration = configuration

    def resolve(
        self,
        environment: Mapping[str, str],
        args: Sequence[str],
        *,
        directory: str,
        configured: Optional[Snapshot] = None,
        specification: Optional[LaunchSpec] = None,
    ) -> Resolution:
        specification = specification or self.configuration
        if specification is None:
            raise ValueError("production Runtime requires explicit typed LaunchSpec")
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
            resolver=partial(
                resolve_typed,
                spec=specification,
                platform=self.policy.platform,
                directory=directory,
                memory=self.policy.memory_budget_bytes,
                configured=configured,
            ),
            observe=adapter,
            emit=self.emit,
            darwin_errors=self.policy.platform == "darwin",
        )
        if value.code != 0:
            raise RuntimeError("native launch preparation failed")
        return value

    def engine(
        self,
        directory: str,
        port: int,
        tier: int,
        cpus: str,
        envs: str,
        args: list[str],
        *,
        configured: Optional[Snapshot] = None,
        specification: Optional[LaunchSpec] = None,
    ) -> Engine:
        if envs:
            raise ValueError("production Engine does not accept legacy environment controls")
        reject_legacy_environment(dict(self.inputs.environment))

        def environment_for(
            environment: Mapping[str, str], _port: int, _envs: str
        ) -> dict[str, str]:
            reject_legacy_environment(environment)
            output = dict(environment)
            return output

        def checked(environment: Mapping[str, str], extra: Sequence[str]) -> Resolution:
            value = self.resolve(
                environment,
                extra,
                directory=directory,
                configured=configured,
                specification=specification,
            )
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
            prepare_environment=environment_for,
            check_port=True,
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
        keeps_history: bool = False,
    ) -> Controller:
        if self.policy.platform != "linux":
            raise ValueError(
                "Darwin adaptive Controller requires the unimplemented M2 placement adapter"
            )
        if self.configuration is None:
            raise ValueError("production Controller requires explicit typed LaunchSpec")
        if keeps_history:
            raise ValueError("typed launches do not configure cross-server prompt persistence")
        specification = self.configuration
        base = parse(
            Path(specification.settings_file).read_text(encoding="utf-8")
            if specification.settings_file
            else None,
            specification.settings_file or "",
            overrides=specification.overrides,
            platform=self.policy.platform,
        )

        def typed_engine(
            directory: str,
            selected_port: int,
            tier: int,
            cpus: str,
            extra: Sequence[str],
            chosen: Selection,
        ) -> Engine:
            suffix = cpus.partition("@")[2]
            selected = replace(specification, threads=chosen.threads)
            if suffix:
                selected = replace(selected, ubatch=int(suffix))
            return self.engine(
                directory,
                selected_port,
                tier,
                cpus,
                "",
                list(extra),
                configured=chosen.settings,
                specification=selected,
            )

        def machine_factory(path: str) -> TypedMachine:
            machine = TypedMachine(
                DirectoryArtifacts(Path(path)),
                base=base,
                threads=specification.threads,
                native_ubatch=specification.ubatch,
                host_threads=host_threads,
            )
            if start not in machine.configs:
                raise ValueError(
                    "start configuration is excluded by typed settings or measured capacity"
                )
            return machine

        return TypedController(
            run,
            out,
            port,
            start,
            args,
            loop,
            window,
            decode,
            horizon,
            machine_factory=machine_factory,
            engine_factory=self.engine,
            configuration=lambda machine, tier, cpus: cast(TypedMachine, machine).selected(
                tier, cpus
            ),
            typed_engine=typed_engine,
            keeps_history=keeps_history,
            post=post,
            effects=LocalEffects({}, stdout),
            session=NativeObservationSession(files=LocalObservations()),
        )
