"""CI host capability report.

Records the runner's platform and accelerator inventory in the test log so CI
failures can be attributed to the executing host. Read-only and skip-safe.
"""

import shutil
import subprocess


def _run(cmd: str) -> str:
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20)
        return (out.stdout + out.stderr).strip()
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"{cmd}: {exc}"


def test_report_host_capabilities(capsys):
    report = ["CI host capability report"]
    checks = [
        "nvidia-smi -L",
        "nvidia-smi --query-gpu=name,uuid,memory.total,memory.free,driver_version --format=csv,noheader",
        "hostname",
        "id",
        "ls -l /dev/nvidia* 2>&1 | head -5",
        "cat /proc/driver/nvidia/version 2>&1 | head -2",
        "lspci 2>/dev/null | grep -iE 'nvidia|3d controller|vga' | head -5",
        "rocm-smi 2>&1 | head -5",
        "hy-smi 2>&1 | head -5",
    ]
    for cmd in checks:
        binary = cmd.split()[0]
        if binary in ("hostname", "id", "ls", "cat", "lspci") or shutil.which(binary):
            report.append(f"$ {cmd}\n{_run(cmd)}")
        else:
            report.append(f"$ {cmd}\n(not present on this host)")
    text = "\n".join(report)
    print(text)
    captured = capsys.readouterr()
    assert "CI host capability report" in captured.out
