#!/usr/bin/env python3
"""Generate private Mac-reader input candidates, not numerical golden outputs.

Reuse the immutable C0 generator with only its expert count changed per model.
The original 16-expert inputs and the repository generator are never edited.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import platform
import sys

GENERATOR_SHA256 = '637c2d8219ca9a506dc2a773700cc074fa8cb689f128839b303021bcfe2c41bc'
SEED = 20261022

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    repo, out = a.repo.resolve(), a.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    generator = repo / 'experiments/c0-store/make-tiny.py'
    if sha(generator) != GENERATOR_SHA256:
        raise ValueError('unreviewed upstream generator')
    sys.path.insert(0, str(repo / 'gguf-py'))
    spec = importlib.util.spec_from_file_location('c0_tiny_input_generator', generator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.np.__version__ != '2.0.2' or (module.D, module.E, module.LAYERS, module.VOCAB) != (256, 16, 2, 259):
        raise ValueError('unexpected generator environment or dimensions')
    out.mkdir(parents=True)
    models = []
    for arch, name, experts in [('deepseek41', 'MXFP4', 384), ('muse-spark', 'Q4_K', 256),
                                ('muse-spark', 'Q5_K', 256)]:
        module.E = experts
        item = module.generate(out, arch, getattr(module.gguf.GGMLQuantizationType, name), SEED)
        # The generator emits absolute alias paths; retain the actual private
        # file hash and separately bind the portable path-normalized semantics.
        alias = out / item['alias']
        lines = alias.read_text().splitlines()
        header, body = lines[0], []
        for line in lines[1:]:
            fields = line.split('\t')
            if len(fields) != 5 or Path(fields[3]) != out / item['gguf']:
                raise ValueError('unexpected alias source')
            fields[3] = '<MODEL_DIR>/' + item['gguf']
            body.append('\t'.join(fields))
        normalized = ('\n'.join([header, *body]) + '\n').encode()
        item['alias_sha256'] = sha(alias)
        item['normalized_alias_sha256'] = hashlib.sha256(normalized).hexdigest()
        item['expert_payload_bytes'] = sum(int(line.split('\t')[2]) for line in lines[1:])
        item['status'] = 'format and pack/alias range validation only; no engine execution; not golden'
        models.append(item)
    sources = {str(f.relative_to(repo)): sha(f) for f in sorted((repo/'gguf-py/gguf').rglob('*.py'))}
    manifest = {'schema': 'mac-reader-tiny-input-candidate-v1', 'golden': False,
                'engine_executed': False, 'source_generator_sha256': GENERATOR_SHA256,
                'wrapper_sha256': sha(Path(__file__).resolve()), 'gguf_python_sources': sources,
                'python': sys.version, 'numpy': module.np.__version__, 'generator_host_arch': platform.machine(),
                'seed': SEED, 'original_expert_count': 16,
                'only_generator_parameter_change': 'E=384 for DeepSeek4.1; E=256 for Muse',
                'vocabulary': 259,
                'models': models, 'total_gguf_and_pack_bytes': sum(m['gguf_bytes']+m['pack_bytes'] for m in models)}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'status': 'format-only inputs prepared', 'total_bytes': manifest['total_gguf_and_pack_bytes'],
                      'models': [{k: m[k] for k in ['architecture','type','experts','range_count','gguf_bytes','pack_bytes','gguf_sha256','pack_sha256']} for m in models]}, indent=2))

if __name__ == '__main__':
    main()
