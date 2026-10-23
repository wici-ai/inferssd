"""The model, placement and installed-framework boundaries used in production."""
import ast
from pathlib import Path
import re

POLICY = "engine/ggml/src/ggml-ssd/place/"
MODEL = "engine/src/ssd/model-adapter"
BACKENDS = {"cpu", "cuda", "metal"}
SOURCES = {".h", ".cpp", ".inc", ".c", ".cu", ".cuh", ".m", ".mm", ".metal"}


def source_issues(path: Path, kind: str, content: str) -> list[str]:
    relative = path.as_posix()
    errors = []
    for number, line in enumerate(content.splitlines(), 1):
        code = line.split("//", 1)[0]
        include = re.search(r'#\s*(?:include|import)\s*[<"]([^">]+)', code)
        issue = None
        if include:
            target = include.group(1)
            if relative.startswith(POLICY) and re.search(
                r"ggml-(?:cpu|cuda|metal)|cpu-sink|cpu-prefill|runtime-services|ggml-backend"
                r"|(?:^|/)(?:io|host|store|stream|worker|thread-pool)(?:/|[.-])"
                r"|^(?:thread|mutex|condition_variable|future|chrono|unistd.h|fcntl.h|pthread.h)$"
                r"|^sys/", target
            ):
                issue = "placement policy includes execution or resource machinery"
            if relative.startswith(MODEL) and re.search(
                r"ggml-(?:cpu|cuda|metal)|cpu-sink|cpu-prefill|runtime-services"
                r"|(?:^|/)(?:io|host|store|stream)/", target
            ):
                issue = "model adaptation includes an executor implementation"
            if kind in BACKENDS and (
                any(f"ggml-{other}" in target for other in BACKENDS - {kind})
                or "model-adapter" in target or re.search(r"(?:^|/)llama", target)
            ):
                issue = "backend includes another backend or model adaptation"
        if (kind in BACKENDS or relative.startswith(POLICY)) and re.search(
            r"\bLLM_ARCH_\w+|\bllama_hparams\b", code
        ):
            issue = "model identity leaks below the model adapter"
        if relative.startswith(POLICY) and re.search(
            r"\bkimi_cpu_sink\b|\bcuda[A-Z]\w*\s*\(", code
        ):
            issue = "placement policy reaches a private sink or device"
        if relative.startswith(MODEL) and re.search(r"\bkimi_cpu_sink\b", code):
            issue = "model adaptation reaches a private sink"
        if issue:
            errors.append(f"{relative}:{number}: {issue}")
    return errors


def framework_issues(path: Path, content: str) -> list[str]:
    tree = ast.parse(content)
    errors = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [module] + [f"{module}.{alias.name}" for alias in node.names]
        if any(name in {"bench", "ggml", "engine.src", "engine.ggml"}
               or name.startswith(("bench.", "engine.src.", "engine.ggml.", "ggml.")) for name in names):
            errors.append(f"{path}:{node.lineno}: framework imports native or benchmark implementation")
    return errors


def check_runtime_boundaries(root: Path) -> list[str]:
    errors = []
    groups = [("policy", root / POLICY), ("model", root / "engine/src/ssd")]
    groups += [(name, root / f"engine/ggml/src/ggml-{name}" / ("ssd" if name != "metal" else ""))
               for name in sorted(BACKENDS)]
    for kind, directory in groups:
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix not in SOURCES:
                continue
            relative = path.relative_to(root)
            if kind == "model" and not relative.as_posix().startswith(MODEL):
                continue
            errors.extend(source_issues(relative, kind, path.read_text()))
    for path in sorted((root / "framework/src/inferssd").rglob("*.py")):
        errors.extend(framework_issues(path.relative_to(root), path.read_text()))
    return errors
