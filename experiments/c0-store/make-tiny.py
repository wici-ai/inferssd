#!/usr/bin/env python3
"""Generate deterministic small GGUF/alias/pack candidates, never golden logits.

Run with the repository gguf-py on PYTHONPATH. The type matrix follows the
current sink guards: DeepSeek-V4.1 MXFP4 and Muse Q4_K/Q5_K.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import gguf
import numpy as np

D = 256
E = 16
LAYERS = 2
VOCAB = 259


def packed_weights(rng: np.random.Generator, kind: gguf.GGMLQuantizationType) -> np.ndarray:
    shape = (E, D, D)
    if kind == gguf.GGMLQuantizationType.MXFP4:
        value = gguf.quantize(rng.normal(0, 0.02, shape).astype(np.float32), kind)
    else:
        # Valid packed blocks with deterministic random codes, rather than a
        # replacement quantizer. gguf-py does not implement Q4_K/Q5_K quantize.
        count = E * D * D // 256
        size = 144 if kind == gguf.GGMLQuantizationType.Q4_K else 176
        blocks = rng.integers(0, 256, (count, size), dtype=np.uint8)
        blocks[:, :2] = np.array([1 / 512], dtype='<f2').view(np.uint8)
        center = 7.5 if kind == gguf.GGMLQuantizationType.Q4_K else 15.5
        blocks[:, 2:4] = np.array([center / 512], dtype='<f2').view(np.uint8)
        blocks[:, 4:12] = 1
        blocks[:, 12:16] = 0x11
        value = blocks.reshape(E, D, size)
    decoded = gguf.dequantize(value, kind)
    if decoded.shape != shape or not np.isfinite(decoded).all() or not np.any(decoded):
        raise ValueError(f'invalid synthetic {kind.name} blocks')
    return value


def metadata(writer: gguf.GGUFWriter, arch: str) -> None:
    writer.add_name(f'C0 candidate {arch}; synthetic weights; not golden')
    writer.add_context_length(128)
    writer.add_embedding_length(D)
    writer.add_block_count(LAYERS)
    writer.add_feed_forward_length(D)
    writer.add_head_count(4)
    writer.add_head_count_kv(1)
    writer.add_key_length(64)
    writer.add_value_length(64)
    writer.add_rope_dimension_count(32)
    writer.add_rope_freq_base(10000.0)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_expert_count(E)
    writer.add_expert_used_count(2)
    writer.add_expert_feed_forward_length(D)
    writer.add_expert_weights_scale(1.0)
    writer.add_expert_weights_norm(True)
    writer.add_uint32(f'{arch}.attention.sliding_window', 32)
    writer.add_custom_alignment(4096)
    writer.add_quantization_version(2)
    writer.add_tokenizer_model('llama')
    writer.add_token_list(['<unk>', '<s>', '</s>'] + [f'<0x{i:02X}>' for i in range(256)])
    writer.add_token_types([2, 3, 3] + [6] * 256)
    writer.add_token_scores([0.0] * VOCAB)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_uint32('tokenizer.ggml.unknown_token_id', 0)
    writer.add_add_bos_token(False)
    writer.add_add_space_prefix(False)
    if arch == 'muse-spark':
        writer.add_float32(f'{arch}.post_norm_epsilon', 1e-6)
        writer.add_rope_freq_base_swa(10000.0)
        writer.add_final_logit_softcapping(30.0)
        writer.add_logit_scale(1.0)
        writer.add_moe_latent_size(D)
        writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SOFTMAX)
        writer.add_key_value(f'{arch}.attention.sliding_window_pattern', [0, 1],
                             gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT32)
        return
    ints = {'attention.q_lora_rank': 64, 'attention.indexer.head_count': 1,
            'attention.indexer.key_length': 32, 'attention.indexer.top_k': 4,
            'attention.output_group_count': 1, 'attention.output_lora_rank': 64,
            'hyper_connection.count': 4, 'hyper_connection.sinkhorn_iterations': 4,
            'expert_shared_count': 1, 'expert_gating_func': 4,
            'engram.head_count': 1, 'engram.key_length': 32, 'engram.max_ngram_size': 2}
    for key, value in ints.items():
        writer.add_uint32(f'{arch}.{key}', value)
    writer.add_float32(f'{arch}.attention.compress_rope_freq_base', 10000.0)
    writer.add_float32(f'{arch}.hyper_connection.epsilon', 1e-6)
    writer.add_swiglu_clamp_exp([7.0, 7.0])
    writer.add_swiglu_clamp_shexp([7.0, 7.0])
    arrays = {'attention.compress_ratios': ([0, 0], gguf.GGUFValueType.UINT32),
              'engram.layer_ids': ([0], gguf.GGUFValueType.UINT32),
              'engram.multipliers': ([1, 3], gguf.GGUFValueType.UINT64),
              'engram.primes': ([17], gguf.GGUFValueType.UINT64),
              'engram.offsets': ([0], gguf.GGUFValueType.UINT64),
              'engram.token_map': (list(range(VOCAB)), gguf.GGUFValueType.INT32)}
    for key, (values, subtype) in arrays.items():
        writer.add_key_value(f'{arch}.{key}', values, gguf.GGUFValueType.ARRAY, subtype)
    writer.add_uint32(f'{arch}.engram.pad_id', 0)


def generate(out: Path, arch: str, kind: gguf.GGMLQuantizationType, seed: int) -> dict[str, object]:
    stem = f'{arch}-{kind.name.lower()}'
    path = out / f'{stem}.gguf'
    if path.exists():
        raise FileExistsError(path)
    rng = np.random.default_rng(seed)
    writer = gguf.GGUFWriter(path, arch)
    metadata(writer, arch)

    def tensor(name: str, ne: tuple[int, ...], fill: float | None = None) -> None:
        shape = tuple(reversed(ne))
        value = rng.normal(0, 0.02, shape).astype(np.float32) if fill is None else np.full(shape, fill, np.float32)
        writer.add_tensor(name, value)

    tensor('token_embd.weight', (D, VOCAB))
    tensor('output.weight', (D, VOCAB))
    tensor('output_norm.weight', (D,), 1.0)
    for layer in range(LAYERS):
        prefix = f'blk.{layer}.'
        tensor(prefix + 'attn_norm.weight', (D,), 1.0)
        tensor(prefix + 'ffn_norm.weight', (D,), 1.0)
        tensor(prefix + 'ffn_gate_inp.weight', (D, E))
        tensor(prefix + 'exp_probs_b.bias', (E,))
        for role in ('up', 'gate', 'down'):
            writer.add_tensor(prefix + f'ffn_{role}_exps.weight', packed_weights(rng, kind), raw_dtype=kind)
        if arch == 'muse-spark':
            for role, width in (('q', D), ('k', 64), ('v', 64)):
                tensor(prefix + f'attn_{role}.weight', (D, width))
            tensor(prefix + 'attn_output.weight', (D, D))
            tensor(prefix + 'attn_gate.weight', (D, D))
            for role in ('q', 'k'):
                tensor(prefix + f'attn_{role}_norm.weight', (64,), 1.0)
            for branch in ('attn', 'ffn'):
                tensor(prefix + f'{branch}_resid_alpha.weight', (D,), 1.0)
                tensor(prefix + f'{branch}_resid_beta.weight', (D,), 0.1)
            tensor(prefix + 'post_ffw_norm.weight', (D,), 1.0)
            for role in ('down', 'up'):
                tensor(prefix + f'ffn_latent_{role}.weight', (D, D))
            for role in ('pre', 'post'):
                tensor(prefix + f'ffn_{role}_exp_norm.weight', (D,), 1.0)
        else:
            tensor(prefix + 'attn_sinks.weight', (4,), 0.0)
            tensor(prefix + 'attn_q_a.weight', (D, 64))
            tensor(prefix + 'attn_q_a_norm.weight', (64,), 1.0)
            tensor(prefix + 'attn_q_b.weight', (64, D))
            tensor(prefix + 'attn_kv.weight', (D, 64))
            tensor(prefix + 'attn_kv_a_norm.weight', (64,), 1.0)
            tensor(prefix + 'attn_output_a.weight', (D, 64))
            tensor(prefix + 'attn_output_b.weight', (64, D))
            for branch in ('attn', 'ffn'):
                tensor(prefix + f'hc_{branch}_fn.weight', (4 * D, 24))
                tensor(prefix + f'hc_{branch}_base.weight', (24,), 0.0)
                tensor(prefix + f'hc_{branch}_scale.weight', (3,), 0.01)
            for role in ('up', 'gate', 'down'):
                tensor(prefix + f'ffn_{role}_shexp.weight', (D, D))
            if layer == 0:
                tensor(prefix + 'engram_embd.weight', (32, 17))
                tensor(prefix + 'engram_wkv.weight', (32, 5 * D))
                tensor(prefix + 'engram_q.weight', (D, 4), 1.0)
                tensor(prefix + 'engram_k.weight', (D, 4), 1.0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    reader = gguf.GGUFReader(path)
    ranges = []
    for item in reader.tensors:
        if '_exps.weight' not in item.name:
            continue
        if tuple(item.shape) != (D, D, E) or item.tensor_type != kind:
            raise ValueError(f'wrong expert tensor shape/type: {item.name}')
        size = int(item.n_bytes) // E
        for expert in range(E):
            ranges.append((item.name, expert, int(item.data_offset) + expert * size, size))
    if len(ranges) != LAYERS * E * 3:
        raise ValueError('missing expert ranges')
    alias = out / f'{stem}.alias.tsv'
    with alias.open('w') as handle:
        handle.write('tensor\texpert_idx\tnbytes\tsource_path\toffset\n')
        for name, expert, offset, size in ranges:
            handle.write(f'{name}\t{expert}\t{size}\t{path.resolve()}\t{offset}\n')
    pack_path = out / f'{stem}.pack'
    data_start = ((40 + 152 * len(ranges) + 4095) // 4096) * 4096
    index = []
    cursor = data_start
    for name, expert, _, size in ranges:
        index.append((name, expert, cursor, size))
        cursor += ((size + 4095) // 4096) * 4096
    with pack_path.open('wb') as target, path.open('rb') as source:
        target.write(struct.pack('<16sIIQQ', b'GGMLMOEPACKv1', 1, 40, len(index), data_start))
        for name, expert, offset, size in index:
            target.write(struct.pack('<128siIQQ', name.encode(), expert, 0, offset, size))
        for (_, _, source_offset, size), (_, _, target_offset, _) in zip(ranges, index):
            source.seek(source_offset)
            payload = source.read(size)
            if len(payload) != size:
                raise ValueError('short GGUF expert range')
            target.seek(target_offset)
            target.write(payload)
        target.truncate(cursor)
    with pack_path.open('rb') as packed, path.open('rb') as source:
        for (_, _, source_offset, size), (_, _, target_offset, _) in zip(ranges, index):
            source.seek(source_offset)
            packed.seek(target_offset)
            if source.read(size) != packed.read(size):
                raise ValueError('pack differs from GGUF source')
    return {'architecture': arch, 'type': kind.name, 'seed': seed, 'experts': E,
            'layers': LAYERS, 'embedding': D, 'range_count': len(ranges),
            'gguf': path.name, 'gguf_bytes': path.stat().st_size,
            'gguf_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'pack': pack_path.name, 'pack_bytes': pack_path.stat().st_size,
            'pack_sha256': hashlib.sha256(pack_path.read_bytes()).hexdigest(),
            'alias': alias.name, 'status': 'format-validated candidate; engine load and sink pending',
            'dsv_compressed_attention': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261022)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result = [generate(args.out, arch, kind, args.seed) for arch, kind in [
        ('deepseek41', gguf.GGMLQuantizationType.MXFP4),
        ('muse-spark', gguf.GGMLQuantizationType.Q4_K),
        ('muse-spark', gguf.GGMLQuantizationType.Q5_K)]]
    manifest = {'golden': False, 'numpy': np.__version__, 'models': result}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
