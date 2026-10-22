#!/usr/bin/env python3
"""Short Linux linked-store feasibility comparison; never a frozen-F golden."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import resource
import socket
import subprocess
import time
import urllib.error
import urllib.request


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + '\n')


def check_stop() -> None:
    if (Path.home() / 'ssdl/STOP').exists():
        raise RuntimeError('device STOP marker exists')


def loaded_files(pid: int) -> list[dict[str, str]]:
    paths = set()
    for line in Path(f'/proc/{pid}/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/'):
            path = Path(fields[5])
            if path.is_file():
                paths.add(path)
    return [{'path': str(path), 'sha256': digest(path)} for path in sorted(paths)]


def probe(server: Path, model: Path, out: Path, mode: str) -> dict:
    check_stop()
    label = model.stem + '-' + mode
    run = out / label
    run.mkdir(parents=True, exist_ok=False)
    env = {name: value for name, value in os.environ.items()
           if not name.startswith(('GGML_', 'LLAMA_', 'SSD_LLAMA_'))
           and name not in ('LD_PRELOAD', 'LD_LIBRARY_PATH')}
    if mode != 'resident':
        env.update({'GGML_KIMI_CPU_SINK': '1', 'GGML_KIMI_CPU_SINK_THREADS': '2',
                    'GGML_KIMI_CPU_SINK_VRAM_BYTES': '0',
                    'GGML_KIMI_CPU_PREFILL_MAX_TOKENS': '16',
                    'GGML_KIMI_CPU_SINK_CACHE_BYTES': '16777216',
                    'GGML_KIMI_CPU_SINK_READY_FIRST': '1',
                    'GGML_KIMI_CPU_SINK_PREFETCH': '0',
                    'GGML_MOE_IO_BACKEND': 'direct', 'GGML_MOE_IO_REQUIRE_DIRECT': '1'})
        key = 'GGML_MOE_EXPERT_PACK' if mode == 'pack' else 'GGML_MOE_EXPERT_GGUF_ALIAS_TSV'
        env[key] = str(model.with_suffix('.pack' if mode == 'pack' else '.alias.tsv'))
    # A rare bind race is a failure, never an implicit retry.
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    argv = [str(server), '-m', str(model), '-c', '128', '-b', '16', '-ub', '16',
            '-ngl', '0', '-t', '2', '-tb', '2', '-np', '1', '--no-warmup',
            '--host', '127.0.0.1', '--port', str(port)]
    request = {'prompt': [3, 4, 5, 6], 'n_predict': 4, 'ignore_eos': True,
               'temperature': 0.0, 'top_k': 1, 'samplers': ['top_k'],
               # This synthetic byte vocabulary can end on an incomplete UTF-8
               # sequence, which the old server fails to serialize with probs.
               # Constrain every arm equally; this is not unconstrained greedy.
               'grammar': 'root ::= [a-z]+',
               'cache_prompt': False, 'return_tokens': True,
               'n_probs': 5, 'post_sampling_probs': False}
    write(run / 'settings.json', {'argv': argv, 'settings': {k: v for k, v in env.items()
          if k.startswith(('GGML_', 'LLAMA_', 'SSD_LLAMA_'))}, 'LD_PRELOAD': None,
          'binary_sha256': digest(server), 'model_sha256': digest(model)})
    write(run / 'request.json', request)
    result = {'label': label, 'mode': mode, 'ready': False, 'completion_ok': False,
              'golden': False, 'status': 'failed'}
    with (run / 'server.log').open('w') as log:
        child = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
        result['pid'] = child.pid
        try:
            until = time.monotonic() + 30
            while child.poll() is None and time.monotonic() < until:
                check_stop()
                try:
                    with urllib.request.urlopen(base + '/health', timeout=1) as response:
                        result['ready'] = response.status == 200
                    if result['ready']:
                        break
                except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException):
                    time.sleep(0.1)
            if result['ready']:
                files = loaded_files(child.pid)
                write(run / 'loaded-files.json', files)
                if any('libggml-cuda' in item['path'] or 'shim' in item['path'] for item in files):
                    raise RuntimeError('unexpected CUDA or preload shim in process maps')
                query = urllib.request.Request(base + '/completion', json.dumps(request).encode(),
                                               {'Content-Type': 'application/json'})
                try:
                    with urllib.request.urlopen(query, timeout=30) as response:
                        payload = json.load(response)
                    write(run / 'response.json', payload)
                    probabilities = payload.get('completion_probabilities', [])
                    result['completion_ok'] = (payload.get('tokens_evaluated') == 4
                        and payload.get('tokens_predicted') == 4 and not payload.get('truncated')
                        and len(payload.get('tokens', [])) == 4
                        and len(probabilities) == 4
                        and [item.get('id') for item in probabilities] == payload['tokens']
                        and all(len(item.get('top_logprobs', [])) == 5 for item in probabilities))
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
            result['pid_absent'] = not Path(f'/proc/{child.pid}').exists()
    log_text = (run / 'server.log').read_text()
    totals = re.findall(r'\[kimi_cpu_sink_total\]([^\n]*)', log_text)
    result['sink_total'] = dict(re.findall(r'(\w+)=([^\s]+)', totals[-1])) if totals else {}
    if mode == 'missing-provider':
        result['status'] = ('expected_failure' if not result['completion_ok']
            and 'CPU sink requires the CPU executor and direct expert-pack resolver; no fallback' in log_text
            and result['final_exit'] != 0 and result['pid_absent'] else 'failed')
    else:
        valid = result['completion_ok'] and result['final_exit'] == 0 and result['pid_absent']
        if mode != 'resident':
            stats = result['sink_total']
            valid = (valid and 'directly_linked=1 initialized=1' in log_text
                and int(stats.get('read_requests', 0)) > 0
                and stats.get('read_errors') == '0' and stats.get('fallback_count') == '0')
        result['status'] = 'passed' if valid else 'failed'
    write(run / 'result.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--baseline-server', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    args.out.mkdir(parents=True, exist_ok=False)
    models = sorted(args.models.glob('*.gguf'))
    if len(models) != 3:
        raise ValueError('expected exactly three generated candidates')
    runs = []
    comparisons = []
    summary = {'golden': False, 'runs': runs, 'comparisons': comparisons}
    try:
        runs.append(probe(args.baseline_server.resolve(), models[0], args.out, 'missing-provider'))
        if runs[-1]['status'] != 'expected_failure':
            raise RuntimeError('missing-provider negative control did not fail as specified')
        for model in models:
            for mode in ('resident', 'alias', 'pack'):
                server = args.baseline_server if mode == 'resident' else args.server
                runs.append(probe(server.resolve(), model, args.out, mode))
                if runs[-1]['status'] != 'passed':
                    raise RuntimeError(f'failed arm: {runs[-1]["label"]}')
            reference = json.loads((args.out / (model.stem + '-resident') / 'response.json').read_text())
            for mode in ('alias', 'pack'):
                candidate = json.loads((args.out / (model.stem + '-' + mode) / 'response.json').read_text())
                comparison = {'model': model.name, 'candidate': mode,
                    'tokens_exact': reference['tokens'] == candidate['tokens'],
                    'full_top5_exact': reference['completion_probabilities'] == candidate['completion_probabilities']}
                comparisons.append(comparison)
        summary['status'] = 'passed' if all(c['tokens_exact'] and c['full_top5_exact'] for c in comparisons) else 'different'
    finally:
        write(args.out / 'summary.json', summary)
        print(json.dumps(summary, indent=2))
    if summary.get('status') != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
