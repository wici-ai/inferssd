#!/usr/bin/env python3
"""Compare one completed GPU32 raw-runner request with recorded F bytes.

Reads artifacts only; never executes a model or generates expected output.
DONE is attested by the original worker completion receipt, not a saved event
transcript. This verifier does not independently attest event ordering,
process/library/source identity, transport coverage, resources or cleanup.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def regular_bytes(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("expected a regular, non-symlink artifact")
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size,
                              value.st_mtime_ns, value.st_ctime_ns)
    if identity(before) != identity(after) or path.is_symlink():
        raise ValueError("artifact changed while reading")
    return data


def object_json(path):
    value = json.loads(regular_bytes(path))
    if not isinstance(value, dict):
        raise ValueError("JSON artifact must be an object")
    return value


def ready(value, expected):
    if not isinstance(value, dict):
        raise ValueError("READY must be an object")
    for key in ("n_vocab", "float_bytes", "effective_n_ctx", "effective_n_batch", "effective_n_ubatch"):
        if type(value.get(key)) is not int:
            raise ValueError("READY dimensions must be integers")
    if value != expected:
        raise ValueError("READY differs from the recorded GPU32 profile")


def logits(data, count, selected):
    if len(data) != count * 4:
        raise ValueError("full-vocabulary float32 byte count differs")
    values = struct.unpack("<" + str(count) + "f", data)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite raw logit")
    if type(selected) is not int or max(range(count), key=values.__getitem__) != selected:
        raise ValueError("selected token is not the lowest-ID raw maximum")


def verify(directory, model, mode):
    directory = Path(directory)
    root = Path(__file__).resolve().parent
    manifest = object_json(root / "manifest.json")
    if manifest.get("protocol") != "gpu32-raw-v1" or manifest.get("requests_per_process") != 1:
        raise ValueError("only the recorded single-request GPU32 profile is accepted")
    cases = [case for case in manifest["cases"]
             if case["model"] == model and case["source_mode"] == mode]
    if len(cases) != 1:
        raise ValueError("unknown model/source case")
    case = cases[0]
    expected_ready = {"event": "ready", "n_vocab": 259, "float_bytes": 4,
                      "byte_order": "little", "effective_n_ctx": 256,
                      "effective_n_batch": 32, "effective_n_ubatch": 32}
    ready(case["ready"], expected_ready)
    ready(object_json(directory / "raw-ready.json"), expected_ready)
    result = object_json(directory / "result.json")
    ready(result.get("raw_ready"), expected_ready)
    if result.get("status") != "passed" or result.get("requests_complete") is not True:
        raise ValueError("missing worker-attested completion of the request/DONE exchange")
    responses = result.get("responses")
    if not isinstance(responses, list) or len(responses) != 1:
        raise ValueError("exactly one completed response is required")
    if {p.name for p in directory.glob("request-*.json")} != {"request-1.json"}:
        raise ValueError("exactly one request receipt is required")
    expected_files = {f"request-1-step-{step}.f32" for step in range(4)}
    if {p.name for p in directory.glob("*.f32")} != expected_files:
        raise ValueError("exactly four raw rows are required")
    receipt = object_json(directory / "request-1.json")
    if type(receipt.get("request")) is not int or receipt["request"] != 1:
        raise ValueError("request identity differs")
    rows = receipt.get("rows")
    if not isinstance(rows, list) or len(rows) != 4 or len(case["rows"]) != 4:
        raise ValueError("four complete row records are required")
    wire_rows = []
    for step, (actual, reference) in enumerate(zip(rows, case["rows"])):
        if not isinstance(actual, dict):
            raise ValueError("row record must be an object")
        geometry = {"step": step, "position": 31 + step, "batch_tokens": 32 if step == 0 else 1}
        for record in (actual, reference):
            for key, value in geometry.items():
                if type(record.get(key)) is not int or record[key] != value:
                    raise ValueError(f"step {step}: {key} differs from GPU32")
            if type(record.get("selected_id")) is not int or record["selected_id"] != case["selected_ids"][step]:
                raise ValueError("selected token differs from the reference")
        reference_name = f"{model}/{mode}/step-{step}.f32"
        if reference.get("file") != reference_name:
            raise ValueError("reference path differs")
        data = regular_bytes(root / reference_name)
        if (reference.get("n_vocab") != 259 or reference.get("dtype") != "IEEE-binary32" or
                reference.get("byte_order") != "little" or reference.get("bytes") != 1036 or
                len(data) != 1036 or hashlib.sha256(data).hexdigest() != reference["sha256"]):
            raise ValueError("reference bytes/shape differ from the manifest")
        logits(data, 259, reference["selected_id"])
        name = f"request-1-step-{step}.f32"
        if actual.get("file") != name:
            raise ValueError("unexpected actual raw filename")
        observed = regular_bytes(directory / name)
        logits(observed, 259, actual["selected_id"])
        if observed != data:
            raise ValueError(f"step {step}: full raw bytes differ")
        wire_rows.append({**reference, "file": name})
    expected_response = {"tokens": case["selected_ids"], "raw_logits": wire_rows}
    if responses[0] != expected_response:
        raise ValueError("worker response receipt differs from the actual complete rows")
    return {"status": "passed-recorded-bytes", "model": model, "source_mode": mode,
            "requests": 1, "rows": 4, "vocabulary": 259, "prompt_tokens": 32,
            "ready_artifact_verified": True, "done_evidence": "worker-completion-receipt",
            "raw_done_event_retained": False, "event_order_independently_attested": False,
            "runtime_identity_coverage_resources_cleanup_attested": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("pack", "alias"), required=True)
    args = parser.parse_args()
    try:
        result = verify(args.actual_dir, args.model, args.mode)
    except (OSError, ValueError, KeyError, TypeError, IndexError, struct.error) as error:
        parser.exit(1, f"GPU32 reference verification failed: {error}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
