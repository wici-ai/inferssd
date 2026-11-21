"""Local file facts plus required recorded host facts; never starts an engine."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from inferssd.launch.resources import GPU_BACKENDS
from inferssd.launch.types import Fact, LaunchInput, Reply
from inferssd.machine import characteristics
from inferssd.strategy import inputs as strategy_inputs


class RecordedArtifacts:
    """Filesystem effects are local; host observations must be supplied explicitly.

    A missing host field fails only when requested. This object does not call
    nvidia-smi, sysctl, systemd, sockets, /proc or a server executable, so the
    model's facts (the engine's) are recorded too.
    """

    def __init__(self, inputs: LaunchInput, host: Mapping[str, object]):
        self.inputs, self.host = inputs, host

    def __call__(self, fact: Fact) -> Reply:
        name, args = fact.name, fact.arguments
        if name == "is_file":
            return Reply(Path(args[0]).is_file())
        if name == "is_executable":
            return Reply(os.access(args[0], os.X_OK))
        if name == "read_text":
            with open(args[0], encoding="utf-8", newline="") as stream:
                return Reply(stream.read())
        if name == "resolved_path":
            path = args[0]
            if path == "~" or path.startswith("~/"):
                path = dict(self.inputs.environment).get("HOME", str(Path.home())) + path[1:]
            return Reply(str(Path(path).expanduser().resolve()))
        if name == "write_exclusive":
            with Path(args[0]).open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(args[1])
            return Reply()
        if name == "cpu_topology" and name not in self.host:
            saved = self.host.get("machine_characteristics")
            if saved is not None:
                data = (
                    characteristics.loads(saved)
                    if isinstance(saved, str)
                    else characteristics.validate(saved)
                )
                return Reply(data["cpu"])
        if (
            name in ("machine_characteristics", "model_drive_key", "engine_gpu_backends")
            and name not in self.host
        ):
            # Absent: no strategy, no drive term in it, or an engine whose GPU backends are unknown.
            return Reply(None)
        if name == "systemd" and name not in self.host:
            # Recordings made before this fact existed came from systemd hosts.
            return Reply(True)
        if name not in self.host:
            raise ValueError(f"Missing recorded host fact: {name}")
        value = self.host[name]
        if name == "cpu_topology":
            return Reply(value)  # The shared RAM thread selector validates this topology.
        boolean = {
            "gpu_available",
            "ss_available",
            "systemd",
            "cgroup_v2",
            "user_manager",
            "memory_controller",
        }
        textual = {
            "filesystem",
            "gpu_memory",
            "gpu_processes",
            "port_busy",
            "scope_suffix",
            "platform_system",
            "platform_machine",
        }
        converted = {"default_threads", "physical_memory", "disk_free"}
        if name == "machine_characteristics":
            # An inferssd-machine/1 object, or its text; the resolver gets canonical text.
            try:
                if isinstance(value, str):
                    return Reply(characteristics.dumps(characteristics.loads(value)))
                return Reply(characteristics.dumps(characteristics.validate(value)))
            except ValueError as error:
                raise ValueError(f"Recorded host fact {name}: {error}") from error
        if name in ("model_facts", "runtime_facts"):
            # Serialised strategy inputs, read back strictly and written out again.
            try:
                if name == "model_facts":
                    facts = strategy_inputs.model_facts_from(value)
                    return Reply(strategy_inputs.model_facts_json(facts))
                runtime = strategy_inputs.runtime_facts_from(value)
                return Reply(strategy_inputs.runtime_facts_json(runtime))
            except ValueError as error:
                raise ValueError(f"Recorded host fact {name}: {error}") from error
        if name == "model_drive_key":
            if value is not None and not isinstance(value, str):
                raise ValueError(f"Recorded host fact {name}: expected text or null")
            return Reply(value)
        if name == "engine_gpu_backends":
            if value is not None and (
                not isinstance(value, list)
                or any(backend not in GPU_BACKENDS for backend in value)
                or len(set(value)) != len(value)
            ):
                raise ValueError(
                    f"Recorded host fact {name}: expected null or distinct names of "
                    + ", ".join(GPU_BACKENDS)
                )
            return Reply(value)
        if name in boolean:
            if not isinstance(value, bool):
                raise ValueError(f"Recorded host fact {name}: expected a JSON boolean")
        elif name == "euid":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"Recorded host fact {name}: expected an integer")
        elif name == "memory_kib":
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 2
                or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
            ):
                raise ValueError(
                    f"Recorded host fact {name}: expected two integers [total, available]"
                )
        elif name in textual:
            if not isinstance(value, str):
                raise ValueError(f"Recorded host fact {name}: expected text")
        elif name in converted:
            if isinstance(value, bool) or not isinstance(value, (str, int)):
                raise ValueError(f"Recorded host fact {name}: expected integer text or an integer")
            if name in ("physical_memory", "disk_free"):
                try:
                    int(value)
                except ValueError as error:
                    raise ValueError(f"Recorded host fact {name}: expected integer text") from error
            value = str(value)
        else:
            raise ValueError(f"Unknown recorded host fact: {name}")
        return Reply(value)
