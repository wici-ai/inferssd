#!/usr/bin/env python3
"""Enforce product ownership and the complete recorded migration, not directory counts."""
import csv
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
DIRECTORIES = {"engine", "framework", "bench", "tests", "scripts", "docs", ".github"}
FILES = {
    ".clang-format", ".clang-tidy", ".dockerignore", ".editorconfig", ".gitattributes",
    ".gitignore", ".gitleaks.toml", ".gitleaksignore", "ARCHITECTURE.md", "AUTHORS",
    "CHANGELOG.md", "CMakeLists.txt", "CMakePresets.json", "CONTRIBUTING.md", "LICENSE",
    "NOTICE", "PATCHES.md", "PROVENANCE.md", "README.md", "SECURITY.md",
}
tracked = set(subprocess.check_output(
    ["git", "-C", str(ROOT), "ls-files"], text=True).splitlines())
errors = []
for path in sorted(tracked):
    first = path.split("/", 1)[0]
    if ("/" in path and first not in DIRECTORIES) or ("/" not in path and path not in FILES):
        errors.append("unexpected root item: " + path)
with (ROOT / "docs/development/path-migration.tsv").open() as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))
new_paths = {row["new"] for row in rows if row["new"] != "-"}
for row in rows:
    old, new = row["old"], row["new"]
    if new != "-" and new not in tracked:
        errors.append("mapped destination is not tracked: " + new)
    if old != new and old in tracked and old not in new_paths and old not in FILES:
        errors.append("retired source path remains: " + old)
    if not row["reason"]:
        errors.append("missing ownership rationale: " + old)
# Native tests belong with their build; retained old commands are test fixtures only.
for path in tracked:
    if path.startswith("tests/") and pathlib.Path(path).suffix in {".c", ".cpp", ".cu"}:
        errors.append("native test outside engine/tests: " + path)
    if path.startswith(("framework/compat/", "engine/tools/adaptive/")):
        errors.append("unregistered legacy production adapter: " + path)
if errors:
    print("\n".join(sorted(set(errors))), file=sys.stderr)
    sys.exit(1)
print(f"Product layout passed: {len(rows)} mapped original paths; {len(tracked)} tracked files")
