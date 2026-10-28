"""Execute only frozen reference launchers against tiny assets and explicit fake effects."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from launch_cases import LINUX_CASES, MAC_CASES

ORACLE = Path(__file__).parent / "fixtures/oracle/launch"


def normalize(value, replacements):
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(str(old), new)
        return value
    if isinstance(value, list):
        return [normalize(x, replacements) for x in value]
    if isinstance(value, tuple):
        return [normalize(x, replacements) for x in value]
    if isinstance(value, dict):
        return {normalize(k, replacements): normalize(v, replacements) for k, v in value.items()}
    return value


def header(path, kind=12):
    def string(value):
        return struct.pack("<Q", len(value)) + value

    tensors = [
        (b"blk.0.ffn_gate_exps.weight", (512, 256, 8), kind),
        (b"blk.0.ffn_up_exps.weight", (512, 256, 8), kind),
        (b"blk.0.ffn_down_exps.weight", (256, 1024, 8), kind),
        (b"blk.0.attn_q.weight", (512, 1024), 8),
        (b"blk.0.attn_odd.weight", (512, 997), 8),
        (b"output.weight", (512, 100), 8),
    ]
    data = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1)
    data += string(b"general.architecture") + struct.pack("<I", 8) + string(b"test")
    for index, (name, shape, tensor_type) in enumerate(tensors):
        data += string(name) + struct.pack(
            f"<I{len(shape)}QIQ", len(shape), *shape, tensor_type, index * (1 << 21) + 96
        )
    path.write_bytes(data)
    os.utime(path, (1700000000, 1700000000))


STUB = r"""#!PYTHON
import json,os,subprocess,sys
from pathlib import Path
name=Path(sys.argv[0]).name
p=Path(os.environ["OBS_CASE"])
h=json.loads((p/"host.json").read_text())
a=sys.argv[1:]
with (p/"calls.jsonl").open("a") as f:f.write(json.dumps([name,a])+"\n")
if name=="nvidia-smi":
 print(h["gpu_mib"] if any("memory.total" in x for x in a) else h["gpu_apps"])
elif name=="ss":print(h["port_busy"])
elif name=="nproc":print(h["cpu_count"])
elif name=="date":print("20260101-120000")
elif name=="systemctl":sys.exit(0)
elif name in ("systemd-run","taskset","llama-server"):
 (p/"FORBIDDEN-EXECUTION").write_text(name);sys.exit(95)
elif name=="stat":print(h["filesystem"])
elif name=="awk" and "/proc/meminfo" in a:print(h["mem_total_kib"],h["mem_available_kib"])
elif name=="awk" and any(x.startswith("/sys/devices/") for x in a):print(h["cpu_count"])
elif name=="grep" and any(x.startswith("/sys/fs/cgroup/") for x in a):sys.exit(0)
elif name=="python3":
 command=next((x for x in a if x in ("index","facts","dense")),None)
 if command==h.get("fail_metadata"):
  print("fixture metadata failure: "+command,file=sys.stderr);sys.exit(9)
 r=subprocess.run([sys.executable,*a],capture_output=True)
 with (p/"metadata.jsonl").open("a") as f:
  f.write(json.dumps({"command":command,"argv":a,"stdout":r.stdout.decode(),
                      "stderr":r.stderr.decode(),"code":r.returncode})+"\n")
 sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);sys.exit(r.returncode)
else:os.execv("/usr/bin/"+name,[name,*a])
"""


def linux_case(case, output=None):
    if sys.platform != "linux":
        raise RuntimeError(
            "The exact reference shell observer requires Linux Bash; "
            "do not simulate Linux on Darwin"
        )
    with tempfile.TemporaryDirectory(prefix="inferssd-launch-reference-") as temporary:
        base = Path(temporary)
        repo = base / "repo"
        d = base / "case"
        shutil.copytree(ORACLE, repo)
        for name in ("stub", "bin", "models", "home"):
            (d / name).mkdir(parents=True)
        model = d / "models/Synthetic-Q4.gguf"
        header(model, case.get("type", 12))
        if case.get("ambiguous"):
            header(d / "models/Synthetic-other.gguf")
        for name in (
            "libkimi-no-mmap-prefetch.so",
            "libkimi-mxfp4-parity.so",
            "libkimi-vram-tier.so",
        ):
            (d / "bin" / name).touch()
        host = {
            "gpu_mib": "32607",
            "gpu_apps": "",
            "port_busy": "",
            "cpu_count": 8,
            "filesystem": "ext4",
            "mem_total_kib": 16777216,
            "mem_available_kib": 14680064,
            "uring": False,
            **case.get("host", {}),
        }
        (d / "host.json").write_text(json.dumps(host))
        for name in (
            "systemctl",
            "systemd-run",
            "taskset",
            "ss",
            "nvidia-smi",
            "nproc",
            "date",
            "stat",
            "awk",
            "grep",
            "python3",
        ):
            p = d / "stub" / name
            p.write_text(STUB.replace("PYTHON", sys.executable, 1))
            p.chmod(0o755)
        p = d / "bin/llama-server"
        p.write_text(STUB.replace("PYTHON", sys.executable, 1))
        p.chmod(0o755)
        if host["uring"]:
            (d / "bin/libggml-cuda.so").write_text("io_uring_queue_init")
        conf = (
            "NAME=synthetic\n"
            "MODEL_GLOB=Synthetic-*.gguf\n"
            "TEMPLATE_KWARGS='{\"a\": 1}' # note\n"
            "SAMPLING=--temp 1.0 --top-k 9\n"
            "ENGINE_ENV=LLAMA_TEST_SWITCH=1 GGML_TEST_GLOB=*\n"
        )
        (d / "model.conf").write_text(conf)
        (d / "selected.idx").write_text(
            "tensor\texpert_idx\tnbytes\tsource_path\toffset\nx\t0\t128\t" + str(model) + "\t96\n"
        )
        (d / "model.pack").write_bytes(b"fixture pack selected without parsing")
        env = {
            "HOME": str(d / "home"),
            "USER": "fixture",
            "PATH": str(d / "stub") + ":/usr/bin:/bin",
            "LC_ALL": "C",
            "PYTHONHASHSEED": "0",
            "MODEL_CONF": str(d / "model.conf"),
            "MODEL_DIR": str(d / "models"),
            "SERVER_BIN": str(d / "bin/llama-server"),
            "CPUSET": "0-7",
            "MEMORY_MAX": "10737418240",
            "LAUNCH_SCOPE": "ssdash-fixture.scope",
            "LAUNCH_DRY_RUN": "1",
            "OBS_CASE": str(d),
        }
        if "settings" in case:
            (d / "settings.env").write_text(case["settings"])
            env["SETTINGS"] = str(d / "settings.env")
        if "dense" in case:
            (d / "dense.tsv").write_text(case["dense"])
            env["DENSE_HOMES"] = str(d / "dense.tsv")
        env.update({k: v.replace("@case", str(d)) for k, v in case.get("env", {}).items()})
        repetitions = []
        for index in range(case.get("repeat", 1)):
            if index == 2:
                os.utime(model, (4102444800, 4102444800))
            for f in (d / "calls.jsonl", d / "metadata.jsonl"):
                f.write_text("")
            r = subprocess.run(
                ["/bin/bash", str(repo / "run-ssd-llama.sh"), *case.get("extra", [])],
                cwd=d,
                env=env,
                capture_output=True,
                text=True,
            )
            assert not (d / "FORBIDDEN-EXECUTION").exists(), "fixture attempted live execution"
            artifacts = {
                str(p.relative_to(d)): p.read_text()
                for p in sorted(d.rglob("*"))
                if p.is_file()
                and (p.name in ("experts.tsv", "experts.json") or p.name.endswith(".alias.tsv"))
            }
            calls = [json.loads(x) for x in (d / "calls.jsonl").read_text().splitlines()]
            metadata = [json.loads(x) for x in (d / "metadata.jsonl").read_text().splitlines()]
            repetitions.append(
                {
                    "stdout": r.stdout,
                    "stderr": r.stderr,
                    "code": r.returncode,
                    "calls": calls,
                    "metadata": metadata,
                    "artifacts": artifacts,
                }
            )
        observed = {
            "case": case,
            "env": env,
            "conf": conf,
            "host": host,
            "uid": os.geteuid(),
            "cgroup_v2": Path("/sys/fs/cgroup/cgroup.controllers").is_file(),
            "runs": repetitions,
        }
        observed = normalize(observed, [(repo, "@repo"), (d, "@case"), (sys.executable, "@python")])
        # UID is an explicitly observed Linux fact; controller path is not hidden.
        if output:
            Path(output).write_text(json.dumps(observed, indent=2, sort_keys=True) + "\n")
        return observed


def mac_case(case):
    with tempfile.TemporaryDirectory(prefix="inferssd-mac-launch-reference-") as temporary:
        base = Path(temporary)
        repo = base / "repo"
        d = base / "case"
        d.mkdir()
        shutil.copytree(ORACLE, repo)
        spec = importlib.util.spec_from_file_location(
            "reference_mac_launcher", repo / "scripts/run-macos-ssd.py"
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
                m.main([profile, mode, *case.get("extra", [])])
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


if __name__ == "__main__":
    platform_name, out = sys.argv[1:]
    cases = LINUX_CASES if platform_name == "linux" else MAC_CASES
    observer = linux_case if platform_name == "linux" else mac_case
    results = [observer(case) for case in cases]
    Path(out).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"platform": platform_name, "cases": len(results), "output": out}))
