#!/usr/bin/env python3
"""Materialize the fixed pre-extraction oracle; never regenerate from new code."""
from pathlib import Path
import shutil
import sys

original = Path(__file__).resolve().parent / "oracle"
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for source in original.iterdir():
    if source.suffix == ".inc" or source.name == "extraction.json":
        shutil.copyfile(source, out / source.name)
