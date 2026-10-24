"""Check every selected pair's actual short runtime before any timed request."""

from pathlib import Path, PurePosixPath
import time

import final_chat
from ci.compare import comparable
from ci.contract import digest, load, save
from ci.selection_contract import linux_stages
from ci.staged_benchmark import verify_stopped
from ci.staged_smoke import smoke
from ci.selected_receipt import receipt


LAUNCH = ("argv", "environment", "cwd", "port", "timeout_s", "request_timeout_s")


def same_launch(functional, performance):
    return all(functional[key] == performance[key] for key in LAUNCH)


def declarations(config, manifest, needs):
    """Require complete, authorized short launches before opening the lease."""
    result = []
    platform = config["paths"]["platform"]
    expected = linux_stages(config, needs, ("base", "candidate")) if platform == "linux" else None
    if platform == "darwin" and manifest.get("selection_resource_policy") != config["host"]["resources"]:
        raise ValueError("preflight requires the actual root Mac resource policy")
    for model in needs["models"]:
        for tier in needs["performance"].get(model, {}).get("tiers", []):
            pair = dict(model=model, tier=tier, sides={})
            for side in ("base", "candidate"):
                data = config["sides"][side]
                spec = data["models"][model]["benchmarks"][tier]
                if not set((*LAUNCH, "os_page_cache_treatment")) <= set(spec):
                    raise ValueError("preflight requires every explicit performance launch field")
                if (not isinstance(spec["argv"], list) or not spec["argv"]
                        or not all(isinstance(arg, str) and arg for arg in spec["argv"])
                        or spec["argv"][0] != str(PurePosixPath(data["paths"]["frozen"]) / data["build"]["binary_relative"])
                        or not isinstance(spec["environment"], dict)
                        or not isinstance(spec["cwd"], str) or not spec["cwd"].startswith("/")
                        or type(spec["port"]) is not int or not 1 <= spec["port"] <= 65535
                        or not isinstance(spec["os_page_cache_treatment"], str)
                        or not spec["os_page_cache_treatment"].strip()):
                    raise ValueError("preflight launch is not the explicit sealed performance policy")
                for key in ("timeout_s", "request_timeout_s"):
                    final_chat.bounded_timeout(spec[key])
                if spec["timeout_s"] <= spec["request_timeout_s"]:
                    raise ValueError("preflight deadline must leave startup time")
                stage = side + "-" + model + "-preflight-" + tier
                if expected is not None and manifest["stages"].get(stage) != expected[stage]:
                    raise ValueError("root manifest does not authorize the exact preflight stage")
                pair["sides"][side] = dict(spec, model=model, tier=tier, revision=data["revision"])
            # These fields are required by the already existing functional run.
            # Reading them now prevents a missing reuse declaration being hidden
            # by a fallback after the lease or a timed request has started.
            functional = config["sides"]["candidate"]["models"][model]["smoke"]
            if not set(LAUNCH) <= set(functional):
                raise ValueError("preflight requires the complete candidate functional launch")
            stage = "candidate-smoke-" + model
            if expected is not None and manifest["stages"].get(stage) != expected[stage]:
                raise ValueError("root manifest does not authorize the candidate functional preflight source")
            result.append(pair)
    return result


def run(session, config, manifest, pairs, builds, functional, output):
    """Reuse an identical functional launch; otherwise run one eight-token smoke."""
    output = Path(output)
    gate = dict(schema_version=1, purpose="selected-performance-preflight", status="failed",
                timed_samples=0, started_monotonic_ns=time.monotonic_ns(), cells=[])
    checked = {}
    try:
        for pair in pairs:
            model, tier = pair["model"], pair["tier"]
            cell = dict(model=model, tier=tier, sides={})
            comparable_pair = {}
            for side, spec in pair["sides"].items():
                if spec["argv"][0] != builds[side]["binary"]["runtime_path"]:
                    raise ValueError("preflight executable differs from the actual sealed build")
                folder = output / "preflight" / model / tier / side
                reuse = side == "candidate" and same_launch(functional[model]["launch"], spec)
                raw = functional[model]["output"] if reuse else folder
                result = functional[model]["result"] if reuse else smoke(
                    session, spec, raw, stage_id=side + "-" + model + "-preflight-" + tier)
                if (result.get("status") != "passed" or result.get("cleanup_confirmed") is not True
                        or result.get("completion_requests") != 1):
                    raise RuntimeError("selected short preflight failed or was unclean; no timing or retry")
                folder.mkdir(parents=True, exist_ok=True)
                observed = receipt(config, manifest, side, model, spec, builds[side], raw,
                                   destination=folder / "comparison-receipt.json")
                cell["sides"][side] = dict(raw=str(Path(raw).relative_to(output)),
                    reused_functional=reuse, receipt_sha256=digest(folder / "comparison-receipt.json"))
                comparable_pair[side] = comparable(observed)
                checked[model, tier, side] = observed
            save(output / "preflight" / model / tier / "comparable.json", comparable_pair)
            if comparable_pair["base"] != comparable_pair["candidate"]:
                raise ValueError("short preflight resources, typed settings, model or loaded image roles differ")
            gate["cells"].append(cell)
        gate.update(status="passed", finished_monotonic_ns=time.monotonic_ns())
    except Exception as error:
        gate["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save(output / "performance-preflight.json", gate)
    return checked


def retained(root, config, needs):
    """The hosted aggregate rejects old/missing or incomplete preflight evidence."""
    root = Path(root)
    gate = load(root / "performance-preflight.json")
    expected = [(model, tier) for model in needs["models"]
                for tier in needs["performance"].get(model, {}).get("tiers", [])]
    if (gate.get("schema_version") != 1 or gate.get("purpose") != "selected-performance-preflight"
            or gate.get("status") != "passed" or gate.get("timed_samples") != 0
            or type(gate.get("started_monotonic_ns")) is not int
            or type(gate.get("finished_monotonic_ns")) is not int
            or not 0 < gate["started_monotonic_ns"] <= gate["finished_monotonic_ns"]
            or [(cell["model"], cell["tier"]) for cell in gate["cells"]] != expected):
        raise ValueError("selected performance preflight is missing, failed or incomplete")
    checked = {}
    for cell in gate["cells"]:
        model, tier = cell["model"], cell["tier"]
        if set(cell["sides"]) != {"base", "candidate"}:
            raise ValueError("preflight must contain both runtime arms")
        pair = {}
        for side, data in cell["sides"].items():
            folder = root / "preflight" / model / tier / side
            reused = data["reused_functional"]
            if type(reused) is not bool or reused and side != "candidate":
                raise ValueError("only an identical candidate functional run can be reused")
            model_data = config["sides"][side]["models"][model]
            if reused and not same_launch(model_data["smoke"], model_data["benchmarks"][tier]):
                raise ValueError("reused functional launch differs from the performance arm")
            raw = root / ("native-" + model) if reused else folder
            if data["raw"] != str(raw.relative_to(root)):
                raise ValueError("preflight raw evidence path differs")
            smoke_result, response = load(raw / "smoke.json"), load(raw / "response.json")
            if (smoke_result.get("status") != "passed" or smoke_result.get("cleanup_confirmed") is not True
                    or smoke_result.get("completion_requests") != 1 or smoke_result.get("output_tokens") != 8
                    or smoke_result.get("model") != model
                    or smoke_result.get("revision") != config["sides"][side]["revision"]
                    or response.get("tokens_predicted") != 8 or response.get("stop_type") != "limit"
                    or response.get("truncated") is not False):
                raise ValueError("retained short preflight request failed or was incomplete")
            verify_stopped(load(raw / "stopped.json"))
            path = folder / "comparison-receipt.json"
            if digest(path) != data["receipt_sha256"]:
                raise ValueError("preflight comparison receipt changed")
            checked[model, tier, side] = load(path)
            pair[side] = comparable(checked[model, tier, side])
        if (load(root / "preflight" / model / tier / "comparable.json") != pair
                or pair["base"] != pair["candidate"]):
            raise ValueError("retained preflight arms are not comparable")
    return checked
