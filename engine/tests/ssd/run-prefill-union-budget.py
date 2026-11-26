# SPDX-License-Identifier: Apache-2.0
"""Model-free native policy and ready-residency checks; all outputs are temporary."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

engine = Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory(prefix="ssd-prefill-union-") as tmp:
    for name, include in [
        ("test-ssd-prefill-union-budget", "ggml-cpu"),
        ("test-ssd-vram-ready", "ggml-cuda/ssd"),
    ]:
        argv = [os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off",
                "-I" + str(engine / "ggml/src" / include), "-I" + str(engine / "ggml/src"),
                "-I" + str(engine / "ggml/include"), str(engine / "tests/ssd" / (name + ".cpp")),
                "-o", str(Path(tmp) / name)]
        built = subprocess.run(argv)
        print(json.dumps({"argv": argv, "returncode": built.returncode}), flush=True)
        built.check_returncode()
        executable = Path(tmp) / name
        executed = subprocess.run([str(executable)])
        print(json.dumps({"argv": [str(executable)], "returncode": executed.returncode,
                          "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest()}), flush=True)
        executed.check_returncode()
