"""Load one declared ownership core: installed package or explicit sealed bundle.

A malformed bundle or unknown API is an error, never a request to fall back to
another implementation. This module does not modify the Python search path.
"""
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

MODULES = ("lease", "process", "guardian")
API_VERSION = 1


def load_core(manifest_path=None):
    explicit = manifest_path or os.environ.get("INFERSSD_NATIVE_CORE_MANIFEST")
    adjacent = Path(__file__).with_name("native-core.json")
    manifest_path = Path(explicit) if explicit else adjacent if adjacent.exists() else None
    loaded, identity = {}, {}
    if manifest_path is None:
        for name in MODULES:
            module = importlib.import_module("inferssd.platform." + name)
            if module.CORE_API_VERSION != API_VERSION:
                raise ValueError("unsupported installed ownership core API")
            path = Path(module.__file__).resolve(strict=True)
            identity[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            loaded[name] = module
        return loaded, {"mode": "installed", "api_version": API_VERSION, "modules": identity}
    manifest_path = manifest_path.resolve(strict=True)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    verified_bytes = {}
    if manifest.get("api_version") != API_VERSION or set(manifest.get("modules", {})) != set(MODULES):
        raise ValueError("unsupported ownership core bundle manifest")
    for name in MODULES:
        item = manifest["modules"][name]
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("ownership core bundle path must be contained")
        path = (manifest_path.parent / relative).resolve(strict=True)
        if manifest_path.parent not in path.parents or not path.is_file():
            raise ValueError("ownership core bundle path escaped")
        source = path.read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        if digest != item["sha256"]:
            raise ValueError("ownership core bundle hash differs: " + name)
        identity[name] = {"path": str(path), "sha256": digest}
        verified_bytes[name] = source
    # Verify every member before importing any code. A private package namespace
    # binds relative imports to these verified modules without changing sys.path.
    package_name = "inferssd_native_core_" + hashlib.sha256(manifest_bytes).hexdigest()
    package = types.ModuleType(package_name)
    package.__path__ = []
    sys.modules[package_name] = package
    for name in MODULES:
        spec = importlib.util.spec_from_file_location(package_name + "." + name, identity[name]["path"])
        if spec is None or spec.loader is None:
            raise ValueError("ownership core bundle module cannot be loaded")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        # Execute exactly the verified bytes; never consult unbound cached pyc
        # or reopen a member after verifying it. Relative imports use this namespace.
        exec(compile(verified_bytes[name], identity[name]["path"], "exec"), module.__dict__)
        if module.CORE_API_VERSION != API_VERSION:
            raise ValueError("unsupported bundled ownership core API")
        loaded[name] = module
    return loaded, {"mode": "bundle", "api_version": API_VERSION,
                    "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                    "modules": identity}


CORE, IDENTITY = load_core()
