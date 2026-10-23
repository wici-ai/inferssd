#!/usr/bin/python3
"""Immutable negative-test payload, run only as the isolated worker identity."""

import json
import os
from pathlib import Path
import socket
import subprocess
import time


def denied(function):
    try:
        function()
        return False
    except (OSError, PermissionError):
        return True


def connect(path):
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(.3)
        connection.connect(path)


def external():
    with socket.socket() as connection:
        connection.settimeout(.3)
        connection.connect(("1.1.1.1", 443))


def write(path):
    descriptor = os.open(path, os.O_WRONLY | os.O_TRUNC)
    os.close(descriptor)


def main():
    spec = json.loads(Path("/probe/spec.json").read_text())
    namespaces = {name: os.readlink("/proc/self/ns/" + name) for name in ("user", "mnt", "net", "pid")}
    groups = os.getgroups()
    gates = {"different_execution_uid": os.getuid() == spec["worker_uid"] and os.getuid() not in (0, spec["controller_uid"]) and
             spec["controller_gid"] not in groups and 0 not in groups}
    for gate, path in (("controller_home_denied", spec["controller_canary"]),
                       ("runner_credentials_denied", spec["credential_canary"])):
        gates[gate] = denied(lambda path=path: Path(path).read_bytes())
    for gate, path in (("control_write_denied", spec["control_canary"]), ("evidence_write_denied", spec["evidence_canary"])):
        gates[gate] = denied(lambda path=path: write(path))
    gates["sudo_denied"] = subprocess.run(["/usr/bin/sudo", "-n", "/usr/bin/true"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3, check=False).returncode != 0 if Path("/usr/bin/sudo").exists() else True
    if spec.get("namespace_launcher"):
        gates["nested_userns_denied"] = subprocess.run(
            ["/usr/bin/unshare", "--user", "--map-root-user", "/usr/bin/true"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3, check=False).returncode != 0
        gates["namespace_launcher_hidden"] = not Path(spec["namespace_launcher"]["path"]).exists()
        mounts = [line.split() for line in Path("/proc/mounts").read_text().splitlines()]
        gates["readonly_private_proc"] = any(row[1] == "/proc" and "ro" in row[3].split(",") for row in mounts)
        gates["kernel_tunables_denied"] = denied(lambda: os.close(os.open("/proc/sys/kernel/hostname", os.O_WRONLY)))
    gates["host_sockets_denied"] = all(denied(lambda path=path: connect(path)) for path in (
        "/run/dbus/system_bus_socket", "/run/systemd/private", "/run/docker.sock",
        "/var/run/docker.sock", "/run/user/" + str(spec["controller_uid"]) + "/bus",
        "/run/inferssd-ci-broker/broker.sock"))
    gates["external_network_denied"] = denied(external)
    gates["user_service_escape_denied"] = subprocess.run(["/usr/bin/systemctl", "--user", "is-system-running"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3, check=False).returncode != 0
    gates["process_signal_denied"] = denied(lambda: os.kill(spec["controller_pid"], 0))
    gates["readonly_model_inputs"] = Path("/models/canary").read_bytes() == b"readonly-model-canary\n" and denied(lambda: write("/models/canary"))
    scope = next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    cgroup = Path("/sys/fs/cgroup") / scope.lstrip("/")
    accounting = {name: (cgroup / name).read_text().strip()
                  for name in ("memory.max", "memory.current", "memory.swap.max", "pids.max")}
    accounting["scope"] = scope
    accounting["write_denied"] = denied(lambda: os.close(os.open(cgroup / "memory.max", os.O_WRONLY)))
    accounting["passed"] = (accounting["memory.max"] == str(spec["memory_max"]) and
                            accounting["memory.swap.max"] == "0" and
                            accounting["pids.max"] == str(spec["tasks_max"]) and
                            int(accounting["memory.current"]) > 0 and accounting["write_denied"])
    # Loopback is usable, while its namespace must differ from the host.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        with socket.create_connection(listener.getsockname(), timeout=1) as sender:
            receiver, _ = listener.accept()
            with receiver:
                sender.sendall(b"loopback")
                loopback = receiver.recv(8) == b"loopback"
    passed = all(gates.values()) and accounting["passed"] and loopback and all(namespaces[k] != spec["host_namespaces"][k] for k in namespaces)
    print(json.dumps({"marker": "inferssd-worker-negative-v1", "passed": passed, "gates": gates,
                      "namespaces": namespaces, "uid": os.getuid(), "gid": os.getgid(), "loopback": loopback,
                      "cgroup_accounting": accounting}), flush=True)
    time.sleep(2)  # give the root observer time to bind actual unit properties
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
