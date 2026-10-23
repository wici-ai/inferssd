#!/usr/bin/env python3
"""Check the source and CMake boundaries of the production SSD modules."""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULES = {
    "core": "engine/ggml/src/ggml-ssd",
    "cpu": "engine/ggml/src/ggml-cpu/ssd",
    "cuda": "engine/ggml/src/ggml-cuda/ssd",
    "runtime": "engine/src/ssd",
    "common": "engine/common/inferssd",
    "server": "engine/tools/server/inferssd",
    "framework": "framework/src/inferssd",
}
errors = []
for name, directory in MODULES.items():
    base = ROOT / directory
    if not base.is_dir():
        errors.append(f"missing {directory}")
        continue
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix not in {".c", ".cpp", ".h", ".cu", ".cuh", ".inc"}:
            continue
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("//", 1)[0]
            issue = None
            if re.search(r"\b(?:dlsym|dlopen)\s*\(|__attribute__\s*\(\(weak\)\)|#pragma\s+weak", code):
                issue = "runtime symbol discovery/weak linkage"
            include = re.search(r'#\s*include\s*[<"]([^">]+)', code)
            if include:
                target = include.group(1)
                if target.endswith((".cpp", ".c", ".cu")):
                    issue = "including a source translation unit"
                if name == "core" and re.search(r"ggml-(?:cpu|cuda)|(?:^|/)llama|server-", target):
                    issue = "core includes a backend or frontend"
                if name == "cpu" and "ggml-cuda" in target:
                    issue = "CPU includes CUDA backend"
                if name == "cuda" and "ggml-cpu" in target:
                    issue = "CUDA includes CPU backend"
            if name in {"core", "cpu", "cuda", "runtime"} and re.search(r"\bgetenv\s*\(", code):
                allowed = path == ROOT / "engine/ggml/src/ggml-ssd/telemetry/debug.cpp" and re.search(
                    r'getenv\("INFERSSD_(?:DEBUG|TRACE_DIR)"\)', code)
                if not allowed:
                    issue = "ambient production configuration"
            if issue:
                errors.append(f"{path.relative_to(ROOT)}:{line_no}: {issue}")
for path in (ROOT / MODULES["core"]).rglob("CMakeLists.txt"):
    content = "\n".join(line.split("#", 1)[0] for line in path.read_text().splitlines())
    for call in re.findall(r"target_link_libraries\s*\((.*?)\)", content, re.S):
        if re.search(r"\b(?:ggml-cpu|ggml-cuda|llama|server-context)\b", call):
            errors.append(f"{path.relative_to(ROOT)}: core links an upper layer")
for backend, other in (("cpu", "cuda"), ("cuda", "cpu")):
    for path in (ROOT / MODULES[backend]).rglob("CMakeLists.txt"):
        for call in re.findall(r"target_link_libraries\s*\((.*?)\)", path.read_text(), re.S):
            if f"ggml-{other}" in call:
                errors.append(f"{path.relative_to(ROOT)}: cross-backend link")
for module in ("store", "io", "host", "place", "stream", "settings", "telemetry"):
    if not (ROOT / MODULES["core"] / module).is_dir():
        errors.append(f"missing core module: {module}")
if errors:
    print("\n".join(errors), file=sys.stderr)
    sys.exit(1)
print("InferSSD production module boundaries passed")
