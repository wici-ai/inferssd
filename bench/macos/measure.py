#!/usr/bin/env python3
"""Run one isolated local configuration; record source, numerical and physical-I/O evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.request


def monotonic():
    # Apple Python 3.9's time.monotonic() has a process-local origin. libc++
    # steady_clock uses CLOCK_MONOTONIC_RAW; use it to align native I/O traces.
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def command(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, timeout=15)


def system_sample(pid):
    devices = plistlib.loads(subprocess.check_output(["ioreg", "-r", "-c", "IOBlockStorageDriver", "-a"], timeout=15))
    disks = {}
    for device in devices:
        children = device.get("IORegistryEntryChildren", [])
        name = children[0].get("IORegistryEntryName", "unknown") if children else "unattached"
        stats = device.get("Statistics", {})
        disks[name] = {key: stats.get(key, 0) for key in ("Bytes (Read)", "Bytes (Write)", "Operations (Read)")}
    vm = command(["vm_stat"])
    page_size = int(re.search(r"page size of (\d+)", vm).group(1))
    memory = {key.strip(): int(value) for key, value in re.findall(r"^([^:\n]+):\s*(\d+)\.", vm, re.M)}
    memory["page_size"] = page_size
    return {"t": monotonic(), "time": time.time(), "disks": disks, "vm": memory,
            "rss_kib": int(command(["ps", "-o", "rss=", "-p", str(pid)]).strip() or "0"),
            "swap": command(["sysctl", "-n", "vm.swapusage"]).strip()}


def parse_stats(log):
    rows = []
    for line in log.splitlines():
        if "[mac_expert_pack] reason=" in line:
            rows.append(dict((key, float(value) if "." in value else int(value))
                             for key, value in re.findall(r"([a-z_]+)=([0-9.]+)", line)))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("dsv", "muse"), required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--alias", type=Path)
    parser.add_argument("--source", choices=("alias", "mmap"), default="alias")
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--read-alignment", type=int, choices=(1, 4096, 16384), default=1)
    parser.add_argument("--cache-mib", type=int, default=8192)
    parser.add_argument("--policy", choices=("lfu", "lru"), default="lfu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--prefill-threads", type=int, default=12)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--pause-pid", type=int, help="Temporarily pause this user's screenpipe process during this run")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    binary = args.binary or repo / "build-mac-pipeline/bin/llama-server"
    model = args.model_file.resolve()
    args.out = args.out.resolve()
    if args.source == "alias" and (args.alias is None or not args.alias.is_file()):
        parser.error("alias source requires an existing --alias index")
    args.out.mkdir(parents=True, exist_ok=False)
    env = {key: value for key, value in os.environ.items() if not key.startswith(("GGML_", "LLAMA_", "SSD_LLAMA_"))}
    overrides = {"LLAMA_MMAP_PREFETCH": "0", "LLAMA_DSV41_TOPK_REUSE": "0", "LLAMA_MAC_COMPACT_METAL": "1",
                 "LLAMA_DROP_MMAP_TENSOR_AFTER_UPLOAD": "1", "LLAMA_MAC_ADAPTIVE_PREFILL": "0",
                 "SSD_LLAMA_MAC_EXPERT_CACHE_MIB": str(args.cache_mib), "SSD_LLAMA_MAC_EXPERT_CACHE_POLICY": args.policy,
                 "SSD_LLAMA_MAC_EXPERT_READ_AHEAD": str(args.depth),
                 "SSD_LLAMA_MAC_EXPERT_READ_ALIGNMENT": str(args.read_alignment)}
    if args.source == "alias":
        overrides["GGML_MOE_EXPERT_GGUF_ALIAS_TSV"] = str(args.alias.resolve())
    if args.trace:
        overrides["SSD_LLAMA_MAC_EXPERT_TRACE"] = str((args.out / "access-trace.txt").resolve())
    env.update(overrides)
    port = 18122
    with socket.socket() as check:
        check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        check.bind(("127.0.0.1", port))
    url = f"http://127.0.0.1:{port}"
    cmd = [str(binary), "-m", str(model), "--mmap", "--no-warmup", "--fit", "off", "--no-repack", "--no-host",
           "-ngl", "99", "--cpu-moe", "--no-op-offload", "-c", "8192", "-b", str(args.batch), "-ub", str(args.batch),
           "-t", str(args.threads), "-tb", str(args.prefill_threads), "--cache-ram", "0", "--parallel", "1",
           "--host", "127.0.0.1", "--port", str(port)]
    save(args.out / "config.json", {"argv": cmd, "env": overrides, "git": command(["git", "-C", str(repo), "rev-parse", "HEAD"]).strip(),
         "status": command(["git", "-C", str(repo), "status", "--short"]), "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
         "cache_definition": "First request has an empty application expert cache; repeats retain it. No OS page-cache purge; physical device counters are recorded.",
         "timing_definition": "read_call_ms sums pread service times across workers; compute_ms is thread-0 work, barrier_ms its synchronization waits. These are not device bandwidth or disjoint whole-request phases."})
    samples = []
    stop_monitor = threading.Event()
    server = None
    paused_pid = None

    def post(route, payload):
        request = urllib.request.Request(url + route, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(request, timeout=1800)

    def interrupted(signum, frame):
        raise InterruptedError(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if args.pause_pid is not None:
            process = command(["ps", "-o", "uid=,stat=,comm=", "-p", str(args.pause_pid)]).strip().split(None, 2)
            if len(process) != 3 or int(process[0]) != os.getuid() or Path(process[2]).name != "screenpipe":
                raise RuntimeError("pause-pid must identify this user's screenpipe process")
            if "T" not in process[1]:
                os.kill(args.pause_pid, signal.SIGSTOP)
                paused_pid = args.pause_pid
        log = (args.out / "server.log").open("w")
        started = monotonic()
        server = subprocess.Popen(cmd, env=env, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
        save(args.out / "pid.json", {"driver": os.getpid(), "server": server.pid})

        def monitor():
            with (args.out / "samples.jsonl").open("w", buffering=1) as output:
                while not stop_monitor.is_set():
                    try:
                        value = system_sample(server.pid)
                        samples.append(value)
                        output.write(json.dumps(value) + "\n")
                    except Exception as exc:
                        output.write(json.dumps({"error": str(exc), "t": monotonic()}) + "\n")
                    stop_monitor.wait(1)
        thread = threading.Thread(target=monitor)
        thread.start()
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"server exited {server.returncode}")
            if monotonic() - started > 600:
                raise TimeoutError("model load exceeded 600 seconds")
            try:
                with urllib.request.urlopen(url + "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                time.sleep(0.5)
        load_s = monotonic() - started
        prompt = args.prompt_file.read_text()
        with post("/tokenize", {"content": prompt, "add_special": True}) as response:
            prompt_ids = json.load(response)["tokens"][:args.prompt_tokens]
        if len(prompt_ids) != args.prompt_tokens:
            raise RuntimeError("prompt token length mismatch")
        save(args.out / "prompt.json", prompt_ids)
        results = []
        for run in range(args.runs):
            before = system_sample(server.pid)
            before_stats = parse_stats((args.out / "server.log").read_text(errors="replace"))
            payload = {"prompt": prompt_ids, "n_predict": args.tokens, "temperature": 0, "seed": 42, "ignore_eos": True,
                       "cache_prompt": False, "return_tokens": True, "n_probs": 5, "stream": True, "timings_per_token": True}
            begin = monotonic()
            first = None
            first_sample = None
            tokens, probabilities, events = [], [], []
            final = None
            with post("/completion", payload) as response:
                for line in response:
                    if not line.startswith(b"data: "):
                        continue
                    text = line[6:].strip()
                    if text == b"[DONE]":
                        break
                    event = json.loads(text)
                    now = monotonic()
                    events.append({"t": now, "event": event})
                    if event.get("tokens") or event.get("completion_probabilities") or event.get("content"):
                        if first is None:
                            first = now
                            first_sample = system_sample(server.pid)
                        tokens.extend(event.get("tokens", []))
                        probabilities.extend(event.get("completion_probabilities", []))
                    if event.get("stop"):
                        final = event
            end = monotonic()
            after = system_sample(server.pid)
            if not tokens:
                tokens = [item["id"] for item in probabilities]
            save(args.out / f"events-{run}.json", events)
            if len(tokens) != args.tokens or final is None:
                raise RuntimeError(f"completion missing tokens or stop: {len(tokens)}")
            result = {"run": run, "cache": "engine-cold" if run == 0 else "engine-warm", "load_s": load_s,
                      "begin": begin, "first": first, "end": end, "wall_s": end - begin, "ttft_s": first - begin,
                      "tokens": tokens, "probabilities": probabilities, "timings": final.get("timings"),
                      "before": before, "after": after, "stats_before": before_stats[-1] if before_stats else {}}
            result["first_sample"] = first_sample
            results.append(result)
            save(args.out / "results.json", results)
            print(json.dumps({key: result[key] for key in ("run", "cache", "wall_s", "ttft_s", "timings")}), flush=True)
        try:
            (args.out / "vmmap-summary.txt").write_text(command(["vmmap", "-summary", str(server.pid)]))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            (args.out / "vmmap-summary.txt").write_text(str(exc) + "\n")
    finally:
        if paused_pid is not None:
            try:
                if Path(command(["ps", "-o", "comm=", "-p", str(paused_pid)]).strip()).name == "screenpipe":
                    os.kill(paused_pid, signal.SIGCONT)
            except (ProcessLookupError, subprocess.CalledProcessError):
                pass
        stop_monitor.set()
        if "thread" in locals():
            thread.join(timeout=20)
        if server and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
        if "log" in locals():
            log.close()
        if (args.out / "server.log").exists():
            save(args.out / "stats.json", parse_stats((args.out / "server.log").read_text(errors="replace")))
        if samples:
            save(args.out / "memory-summary.json", {"peak_rss_kib": max(s["rss_kib"] for s in samples),
                 "first": samples[0], "last": samples[-1],
                 "peak_compressor_pages": max(s["vm"].get("Pages occupied by compressor", 0) for s in samples)})


if __name__ == "__main__":
    main()
