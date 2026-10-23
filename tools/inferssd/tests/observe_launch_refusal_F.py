"""Freeze F's native versus dry-run refusal without executing a launcher.

Only the original die/refuse shell function definitions are loaded. Every
operation in the subprocess is a Bash builtin; no host fact, tool or server is
called. The source remains the already frozen F launcher, not candidate code.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ORACLE = Path(__file__).parent / "fixtures/oracle/launch/run-ssd-llama.sh"
CASES = [
    {"name": "native-port", "dry_run": "0", "message": "Port 8080 is already in use"},
    {"name": "dry-port", "dry_run": "1", "message": "Port 8080 is already in use"},
    {
        "name": "native-multiline",
        "dry_run": "0",
        "message": "The GPU already runs a compute process:\n123 fixture",
    },
    {
        "name": "dry-multiline",
        "dry_run": "1",
        "message": "The GPU already runs a compute process:\n123 fixture",
    },
    {"name": "native-empty-flag", "dry_run": "", "message": "cgroup v2 is required"},
    {
        "name": "native-literal",
        "dry_run": "0",
        "message": "literal $value `text` and 'quoted' 数据",
    },
]


def definitions():
    return "\n".join(
        line
        for line in ORACLE.read_text().splitlines()
        if line.startswith(("die() {", "refuse() {"))
    )


def observe(case):
    script = definitions() + '\ndry_run="$1"\nrefuse "$2"\nprintf "%s\\n" continued\n'
    value = subprocess.run(
        [
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            script,
            "refusal-fixture",
            case["dry_run"],
            case["message"],
        ],
        capture_output=True,
        text=True,
        timeout=5,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
    )
    return dict(case, code=value.returncode, stdout=value.stdout, stderr=value.stderr)


def generate():
    return {
        "source_sha256": hashlib.sha256(ORACLE.read_bytes()).hexdigest(),
        "definitions": definitions(),
        "cases": [observe(case) for case in CASES],
    }


if __name__ == "__main__":
    print(json.dumps(generate(), ensure_ascii=False, indent=2))
