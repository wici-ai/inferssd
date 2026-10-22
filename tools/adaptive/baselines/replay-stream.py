#!/usr/bin/env python3
"""Replay real-text turns like replay-engram.py, streamed, with the time of every decode token.

Each turn appends the assistant reply to the history, so the next turn reuses the cached
conversation. The record of a turn has the server timings plus the decode rates of tokens
1-128, 129-256 and 1-256 measured from the streamed per-token timings.
"""

import argparse
import json
import pathlib
import subprocess
import time
import urllib.request


def rate(milestones, first, last):
    if first not in milestones or last not in milestones:
        return None
    elapsed = (milestones[last] - milestones[first]) / 1e9
    return (last - first) / elapsed if elapsed > 0 else None


def stream_turn(port, history, max_tokens, ignore_eos):
    body = {
        "model": "deepseek-v4",
        "messages": history,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "timings_per_token": True,
    }
    if ignore_eos:
        body["ignore_eos"] = True
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(body, ensure_ascii=False).encode(),
        {"Content-Type": "application/json"},
    )
    milestones, parts = {}, []
    timings = usage = finish = None
    start_ns = time.time_ns()
    with urllib.request.urlopen(request, timeout=7200) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            payload = raw[6:].strip()
            if payload == b"[DONE]":
                break
            event = json.loads(payload)
            now_ns = time.time_ns()
            if event.get("timings"):
                timings = event["timings"]
                n = timings.get("predicted_n")
                if isinstance(n, int) and n >= 1:
                    milestones.setdefault(n, now_ns)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                content = (choice.get("delta") or {}).get("content")
                if content:
                    parts.append(content)
    end_ns = time.time_ns()
    return {
        "start_ns": start_ns,
        "end_ns": end_ns,
        "wall_s": round((end_ns - start_ns) / 1e9, 3),
        "timings": timings or {},
        "usage": usage or {},
        "reply": "".join(parts),
        "finish": finish,
        "ttft_s": (milestones[1] - start_ns) / 1e9 if 1 in milestones else None,
        "decode_first128_tps": rate(milestones, 1, 128),
        "decode_last128_tps": rate(milestones, 128, 256),
        "decode_all_tps": rate(milestones, 1, 256),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18221)
    parser.add_argument("--script", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--logprobs", action="store_true")  # accepted for engram-arm.sh; not used
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--fincore", action="append", default=[])
    args = parser.parse_args()

    history = []
    turns = json.load(open(args.script, encoding="utf-8"))
    with open(args.out, "w", encoding="utf-8") as output:
        for index, entry in enumerate(turns):
            history.append({"role": "user", "content": entry["text"]})
            try:
                record = {"turn": index, **stream_turn(args.port, history, args.max_tokens, args.ignore_eos)}
                history.append({"role": "assistant", "content": record["reply"]})
            except Exception as exc:
                record = {"turn": index, "error": repr(exc)}
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            if args.fincore:
                result = subprocess.run(["fincore", *args.fincore],
                                        text=True, capture_output=True, check=True)
                pathlib.Path(args.out).with_name(f"fincore-turn-{index}.txt").write_text(result.stdout)
            print(json.dumps({k: v for k, v in record.items() if k != "reply"}, ensure_ascii=False), flush=True)
            if "error" in record:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
