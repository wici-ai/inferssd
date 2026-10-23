"""Run the new Darwin policy in the original frozen fixture effect boundary."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from observe_launch_F import ORACLE, normalize


def mac_candidate(case):
    with tempfile.TemporaryDirectory(prefix="ssdash-mac-launch-F-") as temporary:
        base = Path(temporary)
        repo = base / "repo"
        d = base / "case"
        d.mkdir()
        shutil.copytree(ORACLE, repo)
        spec = importlib.util.spec_from_file_location(
            "f_mac_launcher", repo / "scripts/run-macos-ssd.py"
        )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        profile = case.get("profile", "muse")
        mode = case.get("mode", "server")
        model = d / "model.gguf"
        if case.get("split") or case.get("incomplete") or case.get("second_shard"):
            model = d / "model-00001-of-00002.gguf"
            if not case.get("incomplete"):
                (d / "model-00002-of-00002.gguf").touch()
        model.touch()
        if case.get("second_shard"):
            model = d / "model-00002-of-00002.gguf"
        build = d / "build"
        (build / "bin").mkdir(parents=True)
        for name in ("llama-cli", "llama-server"):
            if not case.get("missing_binary"):
                p = build / "bin" / name
                p.write_text("never executed\n")
                p.chmod(0o755)
        template = repo / "models/templates" / m.PROFILES[profile][1]
        template.parent.mkdir(parents=True)
        if not case.get("missing_template"):
            template.touch()
        env = {
            "HOME": str(d / "home"),
            "SSD_LLAMA_MODEL": str(model),
            "SSD_LLAMA_BUILD_DIR": str(build),
            "XDG_CACHE_HOME": str(d / "cache"),
            **case.get("env", {}),
        }
        if case.get("missing_model"):
            env.pop("SSD_LLAMA_MODEL")
        events = []
        execs = []
        stdout = io.StringIO()
        stderr = io.StringIO()
        failure = case.get("failure")

        def system():
            events.append(["platform.system"])
            return case.get("system", "Darwin")

        def machine():
            events.append(["platform.machine"])
            return "arm64"

        def output(argv, **kwargs):
            events.append(["check_output", argv, kwargs])
            if failure == "sysctl":
                raise subprocess.CalledProcessError(3, argv)
            return str(32 * 1024**3)

        def disk(path):
            events.append(["disk_usage", str(path)])
            return SimpleNamespace(free=64 * 1024**3)

        def generate(argv, check):
            events.append(["run", argv, check])
            out = Path(argv[argv.index("--out") + 1])
            summary = Path(argv[argv.index("--summary-json") + 1])
            out.write_text("complete expert index\n")
            if failure == "index":
                raise subprocess.CalledProcessError(9, argv)
            count = len([x for x in argv if x == "--model-glob"])
            summary.write_text(
                json.dumps(
                    {
                        "model_shards": count - (failure == "partial"),
                        "entries": 0 if failure == "empty" else 256,
                    }
                )
            )

        original_replace = m.os.replace

        def replace(source, target):
            events.append(["replace", str(source), str(target)])
            count = len([x for x in events if x[0] == "replace"])
            if failure == "replace" + str(count):
                raise OSError("fixture replace failure")
            original_replace(source, target)

        def execute(binary, argv, environment):
            events.append(["execvpe", binary])
            execs.append([binary, argv, environment])

        class FaultOutput(io.StringIO):
            def write(self, text):
                if failure == "stdout":
                    raise OSError("fixture stdout failure")
                return super().write(text)

        stdout = FaultOutput()

        def candidate_main():
            from ssdash.launch.darwin import resolve_darwin
            from ssdash.launch.prepare import prepare
            from ssdash.launch.recorded import RecordedArtifacts
            from ssdash.launch.types import LaunchInput, Reply

            inputs = LaunchInput(
                str(m.REPO), tuple(env.items()), tuple(case.get("extra", [])), profile, mode
            )
            adapter = RecordedArtifacts(
                inputs,
                {},
                run=lambda argv: m.subprocess.run(argv, check=True),
                replace=m.os.replace,
                python=sys.executable,
            )

            def observe(fact):
                if fact.name == "platform_system":
                    return Reply(m.platform.system())
                if fact.name == "platform_machine":
                    return Reply(m.platform.machine())
                if fact.name == "physical_memory":
                    return Reply(
                        m.subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
                    )
                if fact.name == "disk_free":
                    return Reply(str(m.shutil.disk_usage(Path(fact.arguments[0])).free))
                return adapter(fact)

            def emit(event):
                (sys.stdout if event.channel == "stdout" else sys.stderr).write(event.text)

            result = prepare(
                inputs, resolver=resolve_darwin, observe=observe, emit=emit, darwin_errors=True
            )
            if result.code:
                raise SystemExit(result.code)
            # This is the test's intercepted would-exec recorder, not the product command.
            m.os.execvpe(result.argv[0], list(result.argv), dict(result.effective_environment))

        try:
            with contextlib.ExitStack() as stack:
                for owner, name, value in [
                    (m.platform, "system", system),
                    (m.platform, "machine", machine),
                    (m.subprocess, "check_output", output),
                    (m.subprocess, "run", generate),
                    (m.shutil, "disk_usage", disk),
                    (m.os, "replace", replace),
                    (m.os, "execvpe", execute),
                ]:
                    stack.enter_context(patch.object(owner, name, value))
                stack.enter_context(patch.dict(m.os.environ, env, clear=True))
                stack.enter_context(contextlib.redirect_stdout(stdout))
                stack.enter_context(contextlib.redirect_stderr(stderr))
                candidate_main()
            code = 0
        except SystemExit as error:
            code = error.code
        artifacts = (
            {
                str(p.relative_to(d)): p.read_text()
                for p in sorted((d / "cache").rglob("*"))
                if p.is_file()
            }
            if (d / "cache").exists()
            else {}
        )
        result = {
            "case": case,
            "env": env,
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "code": code,
            "events": events,
            "execs": execs,
            "artifacts": artifacts,
        }
        alias_key = m.hashlib.sha256((profile + "\0" + str(model.resolve())).encode()).hexdigest()[
            :20
        ]
        result = normalize(
            result,
            [
                (repo, "@repo"),
                (d, "@case"),
                (sys.executable, "@python"),
                (alias_key, "<alias-key>"),
            ],
        )

        # Only random private TemporaryDirectory spelling is canonicalized.
        def clean(x):
            if isinstance(x, str):
                return re.sub(r"\.index-[^/\s']+", ".index-<temporary>", x)
            if isinstance(x, list):
                return [clean(y) for y in x]
            if isinstance(x, dict):
                return {k: clean(v) for k, v in x.items()}
            return x

        return clean(result)
