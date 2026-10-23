#!/usr/bin/env python3
"""Prepare a standalone native wrapper bundle with a sealed shared-core identity.

This copies source files only. It never runs a server or contacts a device.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
FILES = ("native_wrapper.py", "native_common.py", "native_core.py", "native_lease.py",
         "native_linux.py", "native_macos.py", "final_chat.py", "stream_chat.py",
         "run_final.py", "run_abba.py", "macos/measurement_images.py")


def bundle(destination, core):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    modules = {}
    for name in ("lease", "process", "guardian"):
        source = Path(core) / (name + ".py")
        relative = "ownership/" + name + ".py"
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        modules[name] = {"path": relative, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    (destination / "native-core.json").write_text(json.dumps({"api_version": 1, "modules": modules}, indent=2) + "\n")
    for name in FILES:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / name, target)
    manifest = {str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(destination.rglob("*")) if p.is_file()}
    (destination / "bundle-manifest.json").write_text(json.dumps({"sha256": manifest}, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True,
                        help="exact ssdash/platform directory from the admitted source or installed wheel")
    args = parser.parse_args()
    bundle(args.output, args.core)
