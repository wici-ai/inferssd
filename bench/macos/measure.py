#!/usr/bin/env python3
"""Run one isolated local configuration; record source, numerical and physical-I/O evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import plistlib
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

from measurement_images import snapshot_images, verify_image_continuity


def monotonic():
    # Apple Python 3.9's time.monotonic() has a process-local origin. libc++
    # steady_clock uses CLOCK_MONOTONIC_RAW; use it to align native I/O traces.
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def command(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, timeout=15)


def system_sample(pid):
    begin = monotonic()
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
    rss = int(command(["ps", "-o", "rss=", "-p", str(pid)]).strip() or "0")
    swap = command(["sysctl", "-n", "vm.swapusage"]).strip()
    end = monotonic()
    return {"t": end, "sample_begin": begin, "sample_end": end, "time": time.time(),
            "disks": disks, "vm": memory, "rss_kib": rss, "swap": swap}


def parse_stats(log):
    rows = []
    for line in log.splitlines():
        if "[mac_expert_pack] reason=" in line:
            rows.append(dict((key, float(value) if "." in value else int(value))
                             for key, value in re.findall(r"([a-z_]+)=([0-9.]+)", line)))
    return rows



class Completion:
    """One HTTP completion, including its own mandatory final event."""
    def __init__(self, expected):
        self.expected = expected
        self.tokens, self.probabilities = [], []
        self.final = None

    def add(self, event):
        if not isinstance(event, dict) or "error" in event or self.final is not None:
            raise ValueError("invalid completion event or data after stop")
        tokens, probabilities = event.get("tokens", []), event.get("completion_probabilities", [])
        if not isinstance(tokens, list) or not isinstance(probabilities, list):
            raise ValueError("tokens and completion_probabilities must be arrays")
        self.tokens.extend(tokens)
        self.probabilities.extend(probabilities)
        if event.get("stop") is True:
            self.final = event

    def validate(self, expected_prompt=None, teacher=False):
        final = self.final
        if final is None or final.get("stop_type") != "limit" or final.get("truncated") is not False:
            raise ValueError("completion lacks an untruncated limit stop")
        if type(final.get("tokens_predicted")) is not int or final["tokens_predicted"] != self.expected:
            raise ValueError("final tokens_predicted mismatch")
        timing = final.get("timings")
        if not isinstance(timing, dict) or type(timing.get("predicted_n")) is not int or timing["predicted_n"] != self.expected:
            raise ValueError("final predicted timing count mismatch")
        prompt_n = timing.get("prompt_n")
        if type(prompt_n) is not int or prompt_n < 0 or (expected_prompt is not None and (prompt_n > expected_prompt if teacher else prompt_n != expected_prompt)):
            raise ValueError("final prompt timing count mismatch")
        for name in ("prompt_ms", "predicted_ms", "prompt_per_second", "predicted_per_second"):
            value = timing.get(name)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid final timing: " + name)
        if len(self.tokens) != self.expected or len(self.probabilities) != self.expected:
            raise ValueError("completion token/probability count mismatch")
        for token, probability in zip(self.tokens, self.probabilities):
            if type(token) is not int or token < 0 or not isinstance(probability, dict) or probability.get("id") != token:
                raise ValueError("completion token/probability ID mismatch")
            top = probability.get("top_logprobs")
            if not isinstance(top, list) or len(top) != 5:
                raise ValueError("completion must contain five raw top_logprobs per token")
            ids = []
            for item in [probability] + top:
                if not isinstance(item, dict) or type(item.get("id")) is not int or item["id"] < 0:
                    raise ValueError("invalid probability token ID")
                value = item.get("logprob")
                if type(value) not in (float, int) or not math.isfinite(value) or value > 1e-6:
                    raise ValueError("invalid raw log probability")
                if not isinstance(item.get("token"), str) or not isinstance(item.get("bytes"), list) or any(type(b) is not int or not 0 <= b <= 255 for b in item["bytes"]):
                    raise ValueError("invalid probability token bytes")
            ids = [item["id"] for item in top]
            if len(set(ids)) != 5:
                raise ValueError("duplicate top_logprobs token IDs")
        return self


@contextmanager
def request_deadline(seconds, signals=None):
    """A total request deadline, in addition to urllib's socket idle timeout."""
    def expired(signum, frame):
        raise TimeoutError("completion total deadline exceeded")
    previous = signal.getsignal(signal.SIGALRM)
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError("measurement requires an unused process real-time timer")
    signal.signal(signal.SIGALRM, signals.handle if signals is not None else expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class TerminationSignals:
    """Defer Python exceptions without passing blocked signal masks to children."""
    def __init__(self):
        self.depth = 0
        self.pending = []

    @staticmethod
    def raise_signal(signum):
        if signum == signal.SIGALRM:
            raise TimeoutError("completion total deadline exceeded")
        raise InterruptedError(f"signal {signum}")

    def handle(self, signum, frame):
        self.pending.append(int(signum))
        if self.depth == 0:
            self.raise_signal(signum)

    @contextmanager
    def defer(self, dispatch=True):
        start = len(self.pending)
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1
            if dispatch and self.depth == 0 and len(self.pending) > start:
                self.raise_signal(self.pending[start])


def process_identity(pid):
    value = subprocess.check_output(["ps", "-o", "uid=,stat=,lstart=,comm=", "-p", str(pid)], text=True, timeout=2).strip().split(None, 7)
    if len(value) != 8:
        raise RuntimeError("cannot establish process identity")
    return {"pid": pid, "uid": int(value[0]), "state": value[1],
            "start": " ".join(value[2:7]), "command": value[7]}


def same_process(left, right):
    return all(left[key] == right[key] for key in ("pid", "uid", "start", "command"))


class PausedScreenpipe:
    def __init__(self, signals=None):
        self.identity = None
        self.signals = signals or TerminationSignals()

    def pause(self, pid):
        with self.signals.defer():
            identity = process_identity(pid)
            if identity["uid"] != os.getuid() or Path(identity["command"]).name != "screenpipe":
                raise RuntimeError("pause-pid must identify this user's screenpipe process")
            if "T" not in identity["state"]:
                # Publish responsibility before SIGSTOP; a failed kill is harmless
                # to resume and cannot leave an unrecorded paused process.
                self.identity = identity
                os.kill(pid, signal.SIGSTOP)

    def resume(self):
        if self.identity is None:
            return
        identity = self.identity
        for attempt in range(2):
            try:
                current = process_identity(identity["pid"])
                if not same_process(identity, current):
                    raise RuntimeError("paused PID identity changed; refusing to signal replacement")
                os.kill(identity["pid"], signal.SIGCONT)
                self.identity = None
                return
            except (OSError, subprocess.SubprocessError):
                if attempt:
                    raise


def clean_resources(paused, server, profiles, stop_monitor, monitor, sampler):
    """Independent cleanup steps: one failure must not skip any other resource."""
    errors, status = [], {}
    def attempt(name, action):
        try:
            status[name] = action()
        except BaseException as exc:
            errors.append({"action": name, "error": repr(exc)})
    attempt("screenpipe_resume", paused.resume)
    stop_monitor.set()
    if monitor is not None:
        def join_monitor():
            monitor.join(timeout=60)
            if monitor.is_alive():
                raise RuntimeError("memory monitor did not stop")
        attempt("monitor_join", join_monitor)
    if sampler is not None:
        attempt("boundary_sampler", lambda: sampler.shutdown(wait=True, cancel_futures=True))
    for i, profiler in enumerate(profiles):
        def finish_profile(proc=profiler):
            try:
                code = proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
                raise RuntimeError("native profile was forcibly stopped")
            if code != 0:
                raise RuntimeError(f"native profile exit {code}")
            return code
        attempt(f"profile_{i}", finish_profile)
    if server is not None:
        def finish_server():
            if server.poll() is not None:
                raise RuntimeError(f"server exited unexpectedly before cleanup: {server.returncode}")
            server.terminate()
            try:
                code = server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
                raise RuntimeError("server required forced kill")
            if code != 0:
                raise RuntimeError(f"server shutdown exit {code}")
            return code
        attempt("server_shutdown", finish_server)
    # Retry an identity-query failure after independently stopping the server.
    if paused.identity is not None:
        attempt("screenpipe_resume_retry", paused.resume)
    return {"actions": status, "errors": errors}


def verify_traces(out, log):
    shutdown = [line for line in log.splitlines() if "[mac_expert_pack] reason=shutdown " in line]
    if len(shutdown) != 1:
        raise ValueError("missing or ambiguous expert-cache shutdown statistics")
    stats = parse_stats(shutdown[0])[0]
    headers = {
        ".requests.tsv": "tensor\texpert\tlogical_bytes\tbegin_ns\thit\tcache_updates",
        ".reads.tsv": "source_path\toffset\tread_bytes\tlogical_bytes\tbegin_ns\tduration_ns",
        ".phases.tsv": "end_ns\trequests\tacquire_us\tread_wait_us\tcompute_us\tbarrier_us\tlookup_us\tpolicy_us\tallocation_us\trelease_us\tlock_wait_us",
    }
    counts, parsed = {}, {}
    for suffix, columns in [("", 2), (".requests.tsv", 6), (".reads.tsv", 6), (".phases.tsv", 11)]:
        path = out / ("access-trace.txt" + suffix)
        data = path.read_text()
        if not data or not data.endswith("\n"):
            raise ValueError(f"empty or truncated trace: {path.name}")
        lines = data.splitlines()
        if suffix and lines[0] != headers[suffix]:
            raise ValueError(f"unexpected trace header: {path.name}")
        rows = lines[1:] if suffix else lines
        parsed[suffix] = [row.split("\t") if suffix else row.split() for row in rows]
        if any(len(row) != columns for row in parsed[suffix]):
            raise ValueError(f"invalid trace rows: {path.name}")
        counts[suffix or "legacy"] = len(rows)
    if counts["legacy"] != stats["requests"] or counts[".requests.tsv"] != stats["requests"] or counts[".reads.tsv"] != stats["read_requests"] or counts[".phases.tsv"] != stats["computed_experts"]:
        raise ValueError("trace row counts do not match shutdown counters")
    if sum(int(row[2]) for row in parsed[".reads.tsv"]) != stats["read_bytes"]:
        raise ValueError("trace read bytes do not match shutdown counters")
    if parsed[""] != [row[:2] for row in parsed[".requests.tsv"]]:
        raise ValueError("legacy and timestamped request traces differ")
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("dsv", "muse"), required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--alias", type=Path)
    parser.add_argument("--source", choices=("alias", "mmap"), default="alias")
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--read-alignment", type=int, choices=(1, 4096, 16384), default=1)
    parser.add_argument("--expert-prefill", action="store_true")
    parser.add_argument("--cache-mib", type=int, default=8192)
    parser.add_argument("--policy", choices=("lfu", "lru"), default="lfu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--prefill-threads", type=int, default=12)
    parser.add_argument("--poll", type=int, help="Override both decode and batch threadpool polling (0..100)")
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--ctx-checkpoints", type=int, help="Override the server context-checkpoint count; 0 disables checkpoints")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--pause-pid", type=int, help="Temporarily pause this user's screenpipe process during this run")
    parser.add_argument("--teacher-reference", type=Path, help="Use the first run's tokens as fixed continuation prefixes; numerical check only")
    parser.add_argument("--request-timeout", type=float, default=1800, help="Total deadline for each completion request, in seconds")
    parser.add_argument("--profile-seconds", type=int, default=0, help="Collect separate native stack profiles; do not use this run as a timing baseline")
    args = parser.parse_args()
    if args.request_timeout <= 0 or min(args.runs, args.tokens, args.prompt_tokens) < 1:
        parser.error("request timeout, runs and token counts must be positive")
    if args.teacher_reference and args.runs != 1:
        parser.error("teacher-forced checks require --runs 1")
    if not 0 <= args.profile_seconds <= 10:
        parser.error("profile-seconds must be between 0 and 10")
    if args.poll is not None and not 0 <= args.poll <= 100:
        parser.error("poll must be between 0 and 100")
    if args.ctx_checkpoints is not None and args.ctx_checkpoints < 0:
        parser.error("ctx-checkpoints must be nonnegative")
    repo = Path(__file__).resolve().parents[2]
    binary = (args.binary or repo / "build-mac-pipeline/bin/inferssd-server").resolve()
    model = args.model_file.resolve()
    args.out = args.out.resolve()
    if args.source == "alias" and (args.alias is None or not args.alias.is_file()):
        parser.error("alias source requires an existing --alias index")
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / "receipt.json", {"status": "running"})
    env = {key: value for key, value in os.environ.items() if not key.startswith(("GGML_", "LLAMA_", "SSD_LLAMA_", "DYLD_"))}
    overrides = {"LLAMA_MMAP_PREFETCH": "0", "LLAMA_DSV41_TOPK_REUSE": "0", "LLAMA_MAC_COMPACT_METAL": "1",
                 "LLAMA_DROP_MMAP_TENSOR_AFTER_UPLOAD": "1", "LLAMA_MAC_ADAPTIVE_PREFILL": "0",
                 "SSD_LLAMA_MAC_EXPERT_CACHE_MIB": str(args.cache_mib), "SSD_LLAMA_MAC_EXPERT_CACHE_POLICY": args.policy,
                 "SSD_LLAMA_MAC_EXPERT_READ_AHEAD": str(args.depth),
                 "SSD_LLAMA_MAC_EXPERT_READ_ALIGNMENT": str(args.read_alignment),
                 "SSD_LLAMA_MAC_EXPERT_PREFILL": "1" if args.expert_prefill else "0"}
    # Build-tree executables may contain absolute LC_RPATH entries. Keep an
    # explicitly selected executable paired with its own shared libraries.
    overrides["DYLD_LIBRARY_PATH"] = str(binary.parent.resolve())
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
    template = "deepseek-ai-DeepSeek-V4.1.jinja" if args.model == "dsv" else "muse-spark.jinja"
    cmd += ["--chat-template-file", str(repo / "models/templates" / template), "--reasoning", "off"]
    if args.model == "dsv":
        cmd += ["-ot", r"blk\.[0-9]+\.engram_embd\.weight=CPU"]
    if args.poll is not None:
        cmd += ["--poll", str(args.poll), "--poll-batch", str(args.poll)]
    if args.ctx_checkpoints is not None:
        cmd += ["--ctx-checkpoints", str(args.ctx_checkpoints)]
    save(args.out / "config.json", {"argv": cmd, "env": overrides, "git": command(["git", "-C", str(repo), "rev-parse", "HEAD"]).strip(),
         "driver": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
         "status": command(["git", "-C", str(repo), "status", "--short"]), "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
         "alias_sha256": hashlib.sha256(args.alias.read_bytes()).hexdigest() if args.source == "alias" else None,
         "prompt_sha256": hashlib.sha256(args.prompt_file.read_bytes()).hexdigest(),
         "shared_library_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted({p.resolve() for p in binary.parent.glob("*.dylib")})},
         "removed_dyld_environment_keys": sorted(key for key in os.environ if key.startswith("DYLD_")),
         "cache_definition": "First request uses a fresh server process, including its normal initialization probes; repeats retain its application cache. No OS page-cache purge; physical device counters are recorded.",
         "timing_definition": "read_call_ms sums pread service times across workers; compute_ms is thread-0 work, barrier_ms its synchronization waits. These are not device bandwidth or disjoint whole-request phases."})
    (args.out / "source.patch").write_text(command(["git", "-C", str(repo), "diff", "HEAD"]))
    samples = []
    sample_errors = []
    stop_monitor = threading.Event()
    server = None
    signals = TerminationSignals()
    paused = PausedScreenpipe(signals)
    profiles = []
    monitor = sampler = None
    validated_requests = []
    results = []
    body_complete = False
    failure = None

    def post(route, payload):
        request = urllib.request.Request(url + route, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(request, timeout=args.request_timeout)

    signal.signal(signal.SIGTERM, signals.handle)
    signal.signal(signal.SIGINT, signals.handle)
    signal.signal(signal.SIGHUP, signals.handle)
    try:
        if args.pause_pid is not None:
            paused.pause(args.pause_pid)
        log = (args.out / "server.log").open("w")
        started = monotonic()
        with signals.defer():
            server = subprocess.Popen(cmd, env=env, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
        save(args.out / "pid.json", {"driver": os.getpid(), "server": server.pid})

        def monitor_memory():
            with (args.out / "samples.jsonl").open("w", buffering=1) as output:
                while not stop_monitor.is_set():
                    try:
                        value = system_sample(server.pid)
                        samples.append(value)
                        output.write(json.dumps(value) + "\n")
                    except Exception as exc:
                        error = {"error": repr(exc), "t": monotonic()}
                        sample_errors.append(error)
                        output.write(json.dumps(error) + "\n")
                    stop_monitor.wait(1)
        with signals.defer():
            monitor = threading.Thread(target=monitor_memory, daemon=True)
            monitor.start()
            sampler = ThreadPoolExecutor(max_workers=1)
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"server exited {server.returncode}")
            if monotonic() - started > 600:
                raise TimeoutError("model load exceeded 600 seconds")
            try:
                with urllib.request.urlopen(url + "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if signals.pending:
                    raise
                time.sleep(0.5)
        load_s = monotonic() - started
        expected_images = json.loads((args.out / "config.json").read_text())

        def image_snapshot(name, require_backends=False):
            if server.poll() is not None:
                raise RuntimeError("server exited before image identity verification")
            value = snapshot_images(server.pid, binary, args.out / name, require_backends=require_backends)
            for item in value["images"]:
                expected = expected_images["binary_sha256"] if item["family"] == "executable" else expected_images["shared_library_sha256"].get(item["path"])
                if item["sha256"] != expected:
                    raise RuntimeError("loaded image differs from selected prelaunch build identity")
            if server.poll() is not None:
                raise RuntimeError("server exited during image identity verification")
            return value

        loaded_before = image_snapshot("images-before")
        prompt = args.prompt_file.read_text()
        with post("/tokenize", {"content": prompt, "add_special": True}) as response:
            prompt_ids = json.load(response)["tokens"][:args.prompt_tokens]
        if len(prompt_ids) != args.prompt_tokens:
            raise RuntimeError("prompt token length mismatch")
        save(args.out / "prompt.json", prompt_ids)
        teacher_ids = None
        if args.teacher_reference:
            reference_bytes = args.teacher_reference.read_bytes()
            reference_prompt_bytes = (args.teacher_reference.parent / "prompt.json").read_bytes()
            reference = json.loads(reference_bytes)
            if not isinstance(reference, list) or not reference or not isinstance(reference[0], dict) or not isinstance(reference[0].get("tokens"), list):
                raise ValueError("invalid teacher reference structure")
            reference_ids = reference[0]["tokens"]
            if any(type(token) is not int or token < 0 for token in reference_ids):
                raise ValueError("teacher reference IDs must be nonnegative integers")
            teacher_ids = reference_ids[:args.tokens]
            if len(teacher_ids) != args.tokens or json.loads(reference_prompt_bytes) != prompt_ids:
                raise ValueError("teacher reference prompt or continuation length mismatch")
            save(args.out / "teacher-reference.json", {"results_path": str(args.teacher_reference.resolve()),
                 "results_sha256": hashlib.sha256(reference_bytes).hexdigest(),
                 "prompt_sha256": hashlib.sha256(reference_prompt_bytes).hexdigest(), "continuation_tokens": teacher_ids})

        def profile(name):
            if args.profile_seconds:
                with signals.defer():
                    profiles.append(subprocess.Popen(["sample", str(server.pid), str(args.profile_seconds), "10", "-file", str(args.out / name)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

        def completion_events(payload):
            with request_deadline(args.request_timeout, signals), post("/completion", payload) as response:
                if not payload["stream"]:
                    yield json.load(response)
                    return
                for line in response:
                    if not line.startswith(b"data: "):
                        continue
                    text = line[6:].strip()
                    if text == b"[DONE]":
                        return
                    yield json.loads(text)

        results = []
        for run in range(args.runs):
            before = system_sample(server.pid)
            before_stats = parse_stats((args.out / "server.log").read_text(errors="replace"))
            payload = {"prompt": prompt_ids, "n_predict": args.tokens, "temperature": 0, "seed": 42, "ignore_eos": True,
                       "cache_prompt": False, "return_tokens": True, "n_probs": 5, "post_sampling_probs": False,
                       "stream": True, "timings_per_token": True}
            begin = monotonic()
            first = last_token_event = None
            first_future = None
            tokens, probabilities, events = [], [], []
            final = None
            profile(f"profile-{run}-prefill.txt")
            try:
                for step in range(args.tokens if teacher_ids is not None else 1):
                    expected = 1 if teacher_ids is not None else args.tokens
                    completion = Completion(expected)
                    if teacher_ids is not None:
                        payload.update(prompt=prompt_ids + teacher_ids[:step], n_predict=1, cache_prompt=True, stream=False)
                    for event in completion_events(payload):
                        now = monotonic()
                        events.append({"t": now, "step": step, "event": event})
                        completion.add(event)
                        if event.get("tokens") or event.get("completion_probabilities"):
                            last_token_event = now
                            if first is None:
                                first = now
                                with signals.defer():
                                    first_future = sampler.submit(system_sample, server.pid)
                                profile(f"profile-{run}-decode.txt")
                    completion.validate(expected_prompt=len(payload["prompt"]), teacher=teacher_ids is not None)
                    validated_requests.append({"run": run, "step": step, "output_tokens": expected,
                                               "stop_type": completion.final["stop_type"]})
                    tokens.extend(completion.tokens)
                    probabilities.extend(completion.probabilities)
                    final = completion.final
            finally:
                end = monotonic()
                save(args.out / f"events-{run}.json", events)
            after = system_sample(server.pid)
            first_sample = first_future.result(timeout=60)
            result = {"run": run, "cache": "fresh-server" if run == 0 else "same-server-repeat", "load_s": load_s,
                      "protocol": "teacher-forced" if teacher_ids is not None else "greedy-generation",
                      "output_tokens": len(tokens), "decode_evaluations": None if teacher_ids is not None else len(tokens) - 1,
                      "performance_eligible": teacher_ids is None and not args.profile_seconds,
                      "client_decode_tokens_per_second": (len(tokens) - 1) / (last_token_event - first) if teacher_ids is None and len(tokens) > 1 and last_token_event > first else None,
                      "last_token_event": last_token_event,
                      "begin": begin, "first": first, "end": end, "wall_s": end - begin, "ttft_s": first - begin,
                      "tokens": tokens, "probabilities": probabilities, "timings": final.get("timings") if teacher_ids is None else None,
                      "before": before, "after": after, "stats_before": before_stats[-1] if before_stats else {}}
            result["first_sample"] = first_sample
            result["first_sample_delay_s"] = [first_sample["sample_begin"] - first, first_sample["sample_end"] - first]
            results.append(result)
            save(args.out / "results.json", results)
            print(json.dumps({key: result[key] for key in ("run", "cache", "wall_s", "ttft_s", "timings")}), flush=True)
        try:
            (args.out / "vmmap-summary.txt").write_text(command(["vmmap", "-summary", str(server.pid)]))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            (args.out / "vmmap-summary.txt").write_text(str(exc) + "\n")
        loaded_after = image_snapshot("images-after", require_backends=True)
        save(args.out / "image-continuity.json", verify_image_continuity(loaded_before, loaded_after))
        body_complete = True
    except BaseException as exc:
        failure = repr(exc)
        raise
    finally:
        with signals.defer(dispatch=False):
            cleanup = clean_resources(paused, server, profiles, stop_monitor, monitor, sampler)
            try:
                if "log" in locals():
                    log.close()
                if (args.out / "server.log").exists():
                    log_text = (args.out / "server.log").read_text(errors="replace")
                    save(args.out / "stats.json", parse_stats(log_text))
                    if args.trace and args.source == "alias":
                        cleanup["trace_counts"] = verify_traces(args.out, log_text)
                if samples:
                    measured = [s for s in samples if any(r["begin"] <= s["sample_begin"] and s["sample_end"] <= r["end"] for r in results)]
                    save(args.out / "memory-summary.json", {"peak_rss_kib": max(s["rss_kib"] for s in samples),
                         "first": samples[0], "last": samples[-1],
                         "request_peak_rss_kib": max((s["rss_kib"] for s in measured), default=None),
                         "request_peak_compressor_pages": max((s["vm"].get("Pages occupied by compressor", 0) for s in measured), default=None),
                         "request_sampling_definition": "One-second samples whose full non-atomic sample interval is inside a recorded request; excludes load and vmmap, and can miss shorter peaks.",
                         "peak_compressor_pages": max(s["vm"].get("Pages occupied by compressor", 0) for s in samples)})
            except BaseException as exc:
                cleanup["errors"].append({"action": "final_evidence", "error": repr(exc)})
            pending = list(signals.pending)
            complete = body_complete and not cleanup["errors"] and not sample_errors and not pending
            receipt = {"status": "complete" if complete else "failed", "failure": failure,
                 "validated_requests": validated_requests, "cleanup": cleanup, "sampling_errors": sample_errors, "pending_signals": pending,
                 "numerical_baseline_comparison": "not performed by this collector; compare tokens and raw top_logprobs separately",
                 "performance_eligible": complete and teacher_ids is None and not args.profile_seconds if "teacher_ids" in locals() else False}
            save(args.out / "receipt.json", receipt)
        if signals.pending:
            complete = False
            receipt.update(status="failed", performance_eligible=False, pending_signals=list(signals.pending))
            with signals.defer(dispatch=False):
                save(args.out / "receipt.json", receipt)
        if body_complete and not complete:
            raise RuntimeError("measurement cleanup or evidence validation failed; see receipt.json")


if __name__ == "__main__":
    main()
