#!/usr/bin/env python3
"""Linux-only dependency smoke, not a golden recorder or performance benchmark."""
from __future__ import annotations

import argparse
import http.client
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import urllib.error
import urllib.request


def probe(server: Path, model: Path, shim: Path, out: Path, use_shim: bool) -> dict[str, object]:
    label = model.stem + ('-store' if use_shim else '-without-store')
    run = out / label
    run.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for name in tuple(env):
        if name.startswith(('GGML_', 'LLAMA_')) or name == 'LD_PRELOAD':
            del env[name]
    env.update({'GGML_KIMI_CPU_SINK': '1', 'GGML_KIMI_CPU_SINK_THREADS': '2',
                'GGML_KIMI_CPU_SINK_VRAM_BYTES': '0',
                'GGML_KIMI_CPU_PREFILL_MAX_TOKENS': '16',
                'GGML_KIMI_CPU_SINK_CACHE_BYTES': '16777216',
                'GGML_KIMI_CPU_SINK_READY_FIRST': '1',
                'GGML_KIMI_CPU_SINK_PREFETCH': '0',
                'GGML_MOE_EXPERT_GGUF_ALIAS_TSV': str(model.with_suffix('.alias.tsv')),
                'GGML_MOE_IO_BACKEND': 'direct', 'GGML_MOE_IO_REQUIRE_DIRECT': '1'})
    if use_shim:
        env['LD_PRELOAD'] = str(shim)
    argv = [str(server), '-m', str(model), '-c', '128', '-b', '16', '-ub', '16',
            '-ngl', '0', '-t', '2', '-tb', '2', '-np', '1', '--no-warmup',
            '--host', '127.0.0.1', '--port', '18343']
    (run / 'settings.json').write_text(json.dumps({'argv': argv, 'settings': {k: v for k, v in env.items()
        if k.startswith(('GGML_', 'LLAMA_')) or k == 'LD_PRELOAD'}}, indent=2) + '\n')
    result: dict[str, object] = {'label': label, 'ready': False, 'completion_ok': False, 'golden': False}
    with (run / 'server.log').open('w') as log:
        child = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
        result['pid'] = child.pid
        try:
            until = time.monotonic() + 30
            while child.poll() is None and time.monotonic() < until:
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18343/health', timeout=1) as response:
                        result['ready'] = response.status == 200
                    if result['ready']:
                        break
                except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException):
                    time.sleep(0.2)
            if result['ready']:
                request = urllib.request.Request('http://127.0.0.1:18343/completion',
                    json.dumps({'prompt': [3, 4, 5, 6], 'n_predict': 4, 'ignore_eos': True,
                                'temperature': 0.0, 'cache_prompt': False}).encode(),
                    {'Content-Type': 'application/json'})
                try:
                    with urllib.request.urlopen(request, timeout=30) as response:
                        payload = json.load(response)
                    (run / 'response.json').write_text(json.dumps(payload, indent=2) + '\n')
                    result['completion_ok'] = payload.get('tokens_predicted') == 4
                    result['tokens_predicted'] = payload.get('tokens_predicted')
                    result['tokens_evaluated'] = payload.get('tokens_evaluated')
                except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as error:
                    result['error'] = str(error)
            result['exit_before_cleanup'] = child.poll()
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
            result['final_exit'] = child.returncode
    (run / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--shim', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--skip-missing-store', action='store_true')
    args = parser.parse_args()
    if (Path.home() / 'ssdl/STOP').exists():
        raise RuntimeError('device STOP marker exists')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    models = sorted(args.models.glob('*.gguf'))
    if len(models) != 3:
        raise ValueError('expected exactly three generated candidates')
    # One no-store arm diagnoses the existing mandatory resolver dependency.
    if not args.skip_missing_store:
        results.append(probe(args.server, models[0], args.shim, args.out, False))
    for model in models:
        results.append(probe(args.server, model, args.shim, args.out, True))
    (args.out / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
