#!/usr/bin/env python3
"""Build one exact revision using the operator's fixed toolchain/profile inputs."""

import argparse
import copy
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import native_common as native
from ci.contract import MODELS, digest, load, save


def build(args):
    config = load(args.config)
    settings = config["build"]
    source, output = args.source.resolve(), args.build.resolve()
    if (
        subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        != args.revision
    ):
        raise ValueError("build source revision mismatch")
    templates = {
        model: load(native.verify_file(config["profiles"][model])["path"])
        for model in (*MODELS, "tiny")
    }
    cmake = settings["cmake"]
    flags, targets = settings["cmake_flags"], settings["targets"]
    if not targets or "llama-server" not in targets:
        raise ValueError(
            "build must include llama-server and the reviewed fixture targets"
        )
    configured = subprocess.run(
        [cmake, "-S", str(source), "-B", str(output), *flags],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(configured.stdout, flush=True)
    if "Manually-specified variables were not used by the project" in configured.stdout:
        raise ValueError("reviewed build flags include an unused/retired CMake control")
    subprocess.run(
        [
            cmake,
            "--build",
            str(output),
            "--parallel",
            str(settings["jobs"]),
            "--target",
            *targets,
        ],
        check=True,
    )
    profiles = {}
    binary = output / settings["binary_relative"]
    images = [output / p for p in settings["image_relatives"]]
    for path in (binary, *images):
        if not path.resolve().is_relative_to(output):
            raise ValueError("build image escaped the owned build directory")
    for model in (*MODELS, "tiny"):
        native.verify_file(config["profiles"][model])
        profile = copy.deepcopy(templates[model])
        template = source / settings["template_relatives"][model]
        if not template.resolve().is_relative_to(source):
            raise ValueError("source template escaped its exact source checkout")
        profile.update(
            revision=args.revision,
            template={"path": str(template.resolve()), "sha256": digest(template)},
            binary={"path": str(binary.resolve()), "sha256": digest(binary)},
            images=[{"path": str(p.resolve()), "sha256": digest(p)} for p in images],
        )
        native.validate_profile(profile)
        if "typed_settings" not in profile:
            raise ValueError("InferSSD CI requires typed native profiles")
        native.verify_inputs(profile)
        profiles[model] = profile
    version = subprocess.check_output(
        [str(binary), "--version"], text=True, stderr=subprocess.STDOUT
    )
    if args.revision[:7] not in version:
        raise ValueError("actual server version does not identify this source revision")
    cache = output / "CMakeCache.txt"
    actual = {}
    for line in cache.read_text().splitlines():
        if "=" in line and not line.startswith(("#", "//")):
            name, value = line.split("=", 1)
            if name.partition(":")[0] in (
                "CMAKE_CXX_COMPILER",
                "CMAKE_C_COMPILER",
                "CMAKE_CUDA_COMPILER",
                "CUDAToolkit_ROOT",
                "CMAKE_CUDA_ARCHITECTURES",
            ):
                actual[name] = value
    toolchain = {
        "cmake": {"path": str(Path(cmake).resolve()), "sha256": digest(cmake)},
        "cache": actual,
    }
    for key, value in actual.items():
        if key.partition(":")[0] in (
            "CMAKE_CXX_COMPILER",
            "CMAKE_C_COMPILER",
            "CMAKE_CUDA_COMPILER",
        ):
            toolchain[key] = {
                "path": str(Path(value).resolve()),
                "sha256": digest(value),
                "version": subprocess.check_output([value, "--version"], text=True),
            }
    save(
        args.receipt,
        {
            "schema_version": 1,
            "purpose": "ci",
            "status": "passed",
            "revision": args.revision,
            "profiles": profiles,
            "toolchain": toolchain,
            "cmake_flags": flags,
            "cmake_cache_sha256": digest(cache),
            "actual_server_version": version,
        },
    )
    args.receipt.with_suffix(".CMakeCache.txt").write_bytes(cache.read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("config", "source", "build", "receipt"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--revision", required=True)
    build(parser.parse_args())
