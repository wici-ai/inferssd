#!/usr/bin/python3
"""Real root-only negative gates; produces no model/performance samples."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract import GATES, Stage, digest, load, protected
from executor import Lease, StageExecutor, quarantine, show
from daemon import FILES
from stage_probe import run as stage_gate


def wait_for(function, seconds=30):
    deadline = time.monotonic() + seconds
    while True:
        result = function()
        if result:
            return result
        if time.monotonic() > deadline:
            raise TimeoutError("negative gate did not reach its required state")
        time.sleep(.05)


def can_claim(config):
    try:
        with Lease(config):
            return True
    except BlockingIOError:
        return False


def hold_child(config, note, connection=None):
    with Lease(config) as lease:
        executor = StageExecutor(config, lease)
        directory, scratch = executor.prepare()
        stage = Stage("inferssd-stage-" + uuid.uuid4().hex + ".service",
                      ("/usr/bin/python3", "-I", "-c", "import time; time.sleep(90)"), (), scratch, timeout=100)
        note.write_text(json.dumps({"unit": stage.unit, "directory": str(directory)}))
        try:
            executor.run(stage, directory, connection)
        except InterruptedError:
            pass


def interrupted_gate(config, directory, hard):
    note = directory / ("hard.json" if hard else "disconnect.json")
    parent, child = socket.socketpair()
    pid = os.fork()
    if pid == 0:
        parent.close()
        try:
            hold_child(config, note, child)
            os._exit(0)
        except BaseException:
            os._exit(1)
    child.close()
    owned = None
    try:
        def active():
            nonlocal owned
            try:
                owned = json.loads(note.read_text())
            except (FileNotFoundError, ValueError):
                return False
            return show(owned["unit"]).get("ActiveState") == "active"
        wait_for(active)
        if can_claim(config):
            raise RuntimeError("original lock was not retained during the owned stage")
        if hard:
            os.kill(pid, signal.SIGKILL)  # exact fresh owned probe controller
        else:
            parent.close()
        _, status = os.waitpid(pid, 0)
        if not hard and status != 0:
            raise RuntimeError("disconnect controller did not cleanly drain")
        path = Path(owned["directory"]) / "guardian.json"
        wait_for(lambda: path.exists())
        receipt = json.loads(path.read_text())
        if receipt.get("cleanup_confirmed") is not True:
            raise RuntimeError("independent guardian did not confirm actual unit cleanup")
        wait_for(lambda: can_claim(config))
        if show(owned["unit"]).get("ActiveState") not in (None, "inactive", "failed"):
            raise RuntimeError("interrupted unit still active")
        return {"passed": True, "unit": owned["unit"], "guardian": receipt, "hard_controller_death": hard}
    finally:
        parent.close()
        # If setup failed, this is still our exact fork; its guardian owns any
        # launched unit and lease. Never scan or signal unrelated processes.
        try:
            os.kill(pid, signal.SIGTERM)
            os.waitpid(pid, 0)
        except (ProcessLookupError, ChildProcessError):
            pass


def stop_gate(config, evidence):
    created = []
    with Lease(config) as lease:
        executor = StageExecutor(config, lease)
        directory, scratch = executor.prepare()
        stage = Stage("inferssd-stage-" + uuid.uuid4().hex + ".service",
                      ("/usr/bin/python3", "-I", "-c", "import time; time.sleep(60)"), (), scratch, timeout=70)
        def stop():
            wait_for(lambda: show(stage.unit).get("ActiveState") == "active")
            reason = "owned-negative-probe-" + uuid.uuid4().hex
            quarantine(config["stop_path"], reason)
            info = Path(config["stop_path"]).stat()
            created.append((reason, info.st_dev, info.st_ino, digest(config["stop_path"])))
        thread = threading.Thread(target=stop)
        thread.start()
        interrupted = False
        try:
            executor.run(stage, directory)
        except InterruptedError:
            interrupted = True
        finally:
            thread.join(timeout=35)
        receipt = json.loads((directory / "receipt.json").read_text())
        try:
            with Lease(config):
                raise RuntimeError("STOP did not block a new admission")
        except RuntimeError as error:
            blocked = "STOP" in str(error)
        passed = interrupted and blocked and len(created) == 1 and receipt.get("cleanup_confirmed") is True
        evidence["stop"] = {"passed": passed, "receipt": receipt}
        # Reset only this test's exact nonce/inode, while still holding the
        # original lease and after confirmed cleanup. Runtime never auto-clears
        # quarantine. Any changed/pre-existing STOP is preserved and fails.
        if passed:
            stop_path = Path(config["stop_path"])
            info = stop_path.stat()
            reason, device, inode, expected_hash = created[0]
            if (info.st_dev, info.st_ino, digest(stop_path)) != (device, inode, expected_hash):
                raise RuntimeError("STOP changed; preserve operator quarantine")
            if json.loads(stop_path.read_text()).get("reason") != reason:
                raise RuntimeError("STOP ownership changed")
            stop_path.unlink()
        if not passed:
            raise RuntimeError("STOP/quarantine negative gate failed")


def main(args):
    if os.geteuid() != 0:
        raise PermissionError("root-only actual isolation probe")
    config = load(args.config)
    install = Path(config["install_root"])
    target = install / "isolation-proof.json"
    if target.exists() or target.is_symlink():
        raise FileExistsError("preserve previous proof; explicit reviewed upgrade required")
    report = {"schema_version": 1, "device": config["device"], "platform": "linux",
              "config_sha256": digest(args.config), "gates": {name: False for name in GATES},
              "formal_samples": 0, "scope": "real CPU namespace/UID/lifecycle and async per-stage broker negatives"}
    root = Path(config["state_root"]) / ("probe-" + uuid.uuid4().hex)
    root.mkdir(mode=0o755)
    controller_home = Path("/var/lib/inferssd-runner")
    canary_name = ".inferssd-negative-" + uuid.uuid4().hex
    canaries = [controller_home / canary_name, controller_home / (canary_name + ".credentials"),
                install / canary_name, root / "evidence-canary"]
    for path in canaries:
        with path.open("xb") as stream:
            stream.write(b"private-negative-canary\n")
        path.chmod(0o600)
    for path in canaries[:2]:
        os.chown(path, config["controller_uid"], config["controller_gid"])
    models = root / "models"
    models.mkdir(mode=0o755)
    (models / "canary").write_bytes(b"readonly-model-canary\n")
    sleeper = subprocess.Popen(["/usr/bin/setpriv", "--reuid=" + str(config["controller_uid"]),
        "--regid=" + str(config["controller_gid"]), "--clear-groups", "/usr/bin/python3", "-I", "-c",
        "import time; time.sleep(180)"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        spec = {"worker_uid": config["worker_uid"], "controller_uid": config["controller_uid"],
                "memory_max": config["memory_max"], "tasks_max": config["tasks_max"],
                "controller_gid": config["controller_gid"], "controller_pid": sleeper.pid,
                "controller_canary": str(canaries[0]), "credential_canary": str(canaries[1]),
                "control_canary": str(canaries[2]), "evidence_canary": str(canaries[3]),
                "host_namespaces": {name: os.readlink("/proc/self/ns/" + name) for name in ("user", "mnt", "net", "pid")}}
        probe = root / "payload"
        probe.mkdir(mode=0o755)
        (probe / "spec.json").write_text(json.dumps(spec))
        (probe / "worker.py").write_bytes((install / "worker_probe.py").read_bytes())
        with Lease(config) as lease:
            executor = StageExecutor(config, lease)
            directory, scratch = executor.prepare()
            stage = Stage("inferssd-stage-" + uuid.uuid4().hex + ".service",
                          ("/usr/bin/python3", "-I", "/probe/worker.py"),
                          ((probe, "/probe"), (models, "/models")), scratch, timeout=30)
            result = executor.run(stage, directory)
            rows = [json.loads(line) for line in (directory / "stdout.log").read_text().splitlines() if line.startswith('{"marker":')]
            if len(rows) != 1 or rows[0].get("passed") is not True or not result["passed"] or "active_unit" not in result:
                raise RuntimeError("actual worker namespace/denial probe failed")
            if any(path.read_bytes() != b"private-negative-canary\n" for path in canaries):
                raise RuntimeError("worker changed trusted canary bytes")
            report["gates"].update(rows[0]["gates"])
            report["gates"]["cleanup_and_reset_verified"] = result["cleanup_confirmed"]
            report["namespace_gate"] = rows[0]
            report["stage"] = result
        report["disconnect"] = interrupted_gate(config, root, False)
        report["controller_death"] = interrupted_gate(config, root, True)
        report["gates"]["controller_disconnect_cleanup"] = True
        stop_gate(config, report)
        report["gates"]["stop_quarantine"] = True
        info = Path(config["lock_path"]).stat()
        report["gates"]["original_lock_inode"] = [info.st_dev, info.st_ino] == config["lock_identity"] and can_claim(config)
        report["async_stage"] = stage_gate(config)
        report["async_stage_gate"] = report["async_stage"]["passed"]
        report["installed_sha256"] = {name: digest(protected(install / name)) for name in FILES}
        if not all(report["gates"].values()):
            raise RuntimeError("not all isolation gates passed")
        with target.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        target.chmod(0o644)
        print(json.dumps({"passed": True, "proof": str(target), "sha256": digest(target), "production_adapter_ready": False, "activation_requires": "root-owned ACTIVE file containing proof SHA256"}))
        return 0
    finally:
        sleeper.terminate()
        sleeper.wait(timeout=10)
        for path in canaries:
            path.unlink(missing_ok=True)
        (root / "attempt.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
