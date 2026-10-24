#!/usr/bin/env python3
"""Run one explicitly admitted release pair; never retry or choose a best sample."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm

METRICS = ("decode_first128_tps", "decode_second128_tps", "decode_tps", "ttft_s", "server_prefill_tps")


def compare(base, candidate):
    if base["prompt_ids"] != candidate["prompt_ids"] or base["prompt_sha256"] != candidate["prompt_sha256"]:
        raise ValueError("authoritative prompt tokens differ")
    rows = {}
    for result in (base, candidate):
        native = result["timings"]
        ms = native.get("prompt_ms")
        if (type(ms) not in (int, float) or not math.isfinite(ms) or ms <= 0
                or native.get("prompt_n") != result["input_tokens"]
                or result["metrics"]["server_prefill_tps"] != result["input_tokens"] * 1000 / ms):
            raise ValueError("prefill must use the actual native prompt timing")
    for key in METRICS:
        before, after = base["metrics"][key], candidate["metrics"][key]
        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in (before, after)):
            raise ValueError("invalid metric: " + key)
        passed = after <= before if key == "ttft_s" else after >= before
        rows[key] = dict(base=before, candidate=after, ratio=after / before, limit_ratio=1.0, passed=passed)
    slower = [key for key in rows if not rows[key]["passed"]]
    return dict(status="inconclusive" if slower else "passed", metrics=rows, observed_slowdowns=slower,
                output_ids_equal=base["tokens"] == candidate["tokens"],
                interpretation="One sample per side: passed means no observed slowdown in this fixed short comparison; it is not statistical proof. No threshold allowance, automatic retry, average, or best-sample selection.")


def run(plan, output):
    order = ["base", "candidate"]
    if plan.get("attempt") == "reverse-diagnostic":
        if not plan.get("hypothesis") or not plan.get("initial_pair"):
            raise ValueError("one reverse diagnostic requires the retained initial pair and a specific hypothesis")
        order.reverse()
    elif plan.get("attempt") != "initial":
        raise ValueError("only initial or one explicit reverse diagnostic is allowed")
    configs = {side: json.loads(Path(plan[side]).read_text()) for side in order}
    common_keys = ("effective_contract", "shards", "template", "alias", "policy", "source_mode", "h2d_limit_bytes_per_second")
    if any(configs["base"].get(k) != configs["candidate"].get(k) for k in common_keys):
        raise ValueError("reviewed effective resources, source, template, or model differ")
    if not configs["base"].get("effective_contract"):
        raise ValueError("explicit reviewed semantic comparison contract is required")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(purpose="release-comparison", full_ci_acceptance=False, formal_final=False,
                  attempt=plan["attempt"], order=order, samples_per_side=1, status="failed", arms={})
    arm.common.save(output / "plan.json", dict(plan=plan, profiles=configs))
    policy = configs["base"]["policy"]
    try:
        with Path(policy["lock_path"]).open("r+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for side in order:
                if Path(policy["stop_path"]).exists():
                    raise InterruptedError("device STOP exists")
                value = arm.run(configs[side], output / side, lock.fileno())
                report["arms"][side] = value
                arm.common.save(output / "pair.json", report)
                if value["status"] != "passed" or value.get("cleanup_confirmed") is not True:
                    raise RuntimeError(side + " measurement or cleanup failed; no automatic repeat")
            a, b = [json.loads((output / side / "runtime-before.json").read_text()) for side in ("base", "candidate")]
            if any(a[k] != b[k] for k in ("boot_id", "cpus", "limits")):
                raise ValueError("actual host, CPU affinity, or process limits differ")
            report.update(compare(report["arms"]["base"]["result"], report["arms"]["candidate"]["result"]))
            report["evidence_valid"] = True
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    arm.common.save(output / "pair.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = run(json.loads(args.plan.read_text()), args.output.resolve())
    print(json.dumps({k: v for k, v in value.items() if k != "arms"}, sort_keys=True))
    raise SystemExit(0 if value["status"] == "passed" else 1)
