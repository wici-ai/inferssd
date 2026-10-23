#!/usr/bin/env python3
"""expert_source.py on a synthetic GGUF header (no model, no drive read): the expert index, the dense weights it
records, the blocks a drive-homed matrix is read in, and the commands that read only expert rows.
usage: python3 framework/tests/compat/test_expert_source.py   (unittest; exit 1 on a failure)"""
import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../fixtures/adaptive'))
import expert_source  # noqa: E402

Q8_0, F32, Q4_K = 8, 0, 12


def gguf(path, tensors):
    """A GGUF v3 header with the architecture only and TENSORS [(name, shape, type)], data offsets in order (32-byte
    aligned, as the converters write them); no tensor data."""
    s = lambda b: struct.pack('<Q', len(b)) + b                       # noqa: E731
    sizes = {Q8_0: (32, 34), F32: (1, 4), Q4_K: (256, 144)}
    out = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), 1) + s(b'general.architecture') + struct.pack('<I', 8) + \
        s(b'testarch')
    off = 0
    for name, shape, t in tensors:
        out += s(name.encode()) + struct.pack(f'<I{len(shape)}QIQ', len(shape), *shape, t, off)
        n = 1
        for d in shape:
            n *= d
        block, size = sizes[t]
        off += -(-(n // block * size) // 32) * 32
    with open(path, 'wb') as f:
        f.write(out)


class Index(unittest.TestCase):
    # two layers: experts of 4 x 512 Q4_K (8 experts, the down slice the largest), and dense matrices of several
    # sizes: one in blocks of whole rows, one whose rows admit no split within [n0, 2 n0], a vector, the embedding
    TENSORS = [('token_embd.weight', (512, 1000), Q8_0), ('output_norm.weight', (512,), F32)] + [
        t for layer in (0, 1) for t in (
            (f'blk.{layer}.attn_q.weight', (512, 1024), Q8_0),         # 557056 B: 2 blocks of 512 rows
            (f'blk.{layer}.attn_norm.weight', (512,), F32),
            (f'blk.{layer}.attn_odd.weight', (512, 997), Q8_0),        # 997 rows, prime: no split
            (f'blk.{layer}.ffn_gate_exps.weight', (512, 256, 8), Q4_K),
            (f'blk.{layer}.ffn_up_exps.weight', (512, 256, 8), Q4_K),
            (f'blk.{layer}.ffn_down_exps.weight', (256, 1024, 8), Q4_K))] + [('output.weight', (512, 1000), Q8_0)]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model = os.path.join(self.tmp.name, 'model.gguf')
        gguf(self.model, self.TENSORS)
        self.d = os.path.join(self.tmp.name, 'idx')

    def tearDown(self):
        self.tmp.cleanup()

    def index(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            expert_source.index(self.model, None, self.d)
        return json.load(open(f'{self.d}/experts.json'))

    def test_the_lane_timings_arguments(self):
        # the probe's lane timing at the experts' shape (up: latent x hidden) and each role's type
        self.index()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            expert_source.lane_args(self.d)
        self.assertEqual(out.getvalue().strip(), '--latent 512 --hidden 256 --up Q4_K --down Q4_K')

    def test_the_dense_weights_and_their_blocks(self):
        info = self.index()
        role = 1024 * 256 // 256 * 144                                  # the down slice: 147456 B
        self.assertEqual(info['bytes_per_role'], {'down': role, 'gate': 512 * 256 // 256 * 144, 'up': 73728})
        dense = {x['name']: x for x in info['dense']}
        self.assertEqual(list(dense)[:3], ['token_embd.weight', 'output_norm.weight', 'blk.0.attn_q.weight'])
        self.assertNotIn('blk.0.ffn_down_exps.weight', dense)
        q = dense['blk.1.attn_q.weight']
        self.assertEqual((q['layer'], q['type'], q['bytes'], q['rows'], q['row_bytes']), (1, 'Q8_0', 557056, 1024, 544))
        # n0 = ceil(557056 / 147456) = 4 blocks of 256 rows (139264 B each, at most the largest role slice)
        self.assertEqual(q['n_blocks'], 4)
        self.assertLessEqual(q['bytes'] // q['n_blocks'], role)
        self.assertEqual(q['bytes'] % q['n_blocks'], 0)
        self.assertIsNone(dense['blk.0.attn_odd.weight']['n_blocks'])  # 997 rows: none of 4..8 divides them
        self.assertIsNone(dense['blk.0.attn_norm.weight']['n_blocks'])  # not a matrix
        self.assertEqual((dense['output.weight']['layer'], dense['output.weight']['rows']), (None, 1000))
        self.assertEqual(len(info['dense']), len(self.TENSORS) - 6)

    def test_the_block_rule(self):
        # Muse's attention matrices: 8 blocks of 1024 rows; its latents 4; k and v 1; its output head 154 of 1312 rows
        S = 11534336
        self.assertEqual(expert_source.blocks(8192, 71303168, S), 8)
        self.assertEqual(expert_source.blocks(4096, 35651584, S), 4)
        self.assertEqual(expert_source.blocks(1024, 8912896, S), 1)
        self.assertEqual(expert_source.blocks(202048, 1758625792, S), 154)
        self.assertIsNone(expert_source.blocks(997, 997 * 544, 139264))

    def test_expert_rows_alone_for_the_other_commands(self):
        info = self.index()
        rows = open(f'{self.d}/experts.tsv').read().splitlines()
        self.assertEqual(len(rows), 1 + 2 * 3 * 8)                     # the header, then expert role slices only
        self.assertTrue(all('_exps.weight' in r for r in rows[1:]))
        out = os.path.join(self.tmp.name, 'gate.iolog')
        expert_source.iolog(self.d, 5, 1, out)
        self.assertEqual(sum(1 for x in open(out) if ' read ' in x), 15)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            expert_source.facts(self.model, info['alias'])
        self.assertEqual(f.getvalue(), "expert_types='Q4_K'\nalias_unaligned=48\n")

    def dense(self, homes):
        hp, out = os.path.join(self.tmp.name, 'homes.tsv'), os.path.join(self.tmp.name, 'dense.tsv')
        with open(hp, 'w') as f:
            f.write(homes)
        g = io.StringIO()
        with contextlib.redirect_stdout(g):
            expert_source.dense(self.model, hp, out)
        with open(out) as f:
            return f.read().splitlines(), dict(x.split('=') for x in g.getvalue().split())

    def test_the_drive_homes_alias_rows(self):
        # the drive's rows only, block by block: equal blocks that tile each matrix exactly, each at most the largest
        # role slice; the RAM's rows counted, not indexed; the engine's stage: two halves of 4 slots, each a block of
        # 139264 B widened by a page and rounded up to one (143360 B)
        rows, totals = self.dense('# homes\nblk.1.attn_q.weight\tdrive\nblk.0.attn_q.weight\tdrive\n'
                                  'output.weight\tram\n')
        self.assertEqual(rows[0], 'tensor\texpert_idx\tnbytes\tsource_path\toffset')
        self.assertEqual(totals, {'dense_drive': '2', 'dense_drive_bytes': str(2 * 557056), 'dense_ram': '1',
                                  'dense_ram_bytes': str(1000 * 544), 'dense_stage_bytes': str(2 * 4 * 143360)})
        info = self.index()
        role = max(info['bytes_per_role'].values())
        heads = {t.name: t for h in expert_source.gguf_header.read_model(self.model) for t in h.tensors}
        by = {}
        for r in rows[1:]:
            t, e, nb, p, o = r.split('\t')
            by.setdefault(t, []).append((int(e), int(nb), p, int(o)))
        self.assertEqual(sorted(by), ['blk.0.attn_q.weight', 'blk.1.attn_q.weight'])
        for name, blocks in by.items():
            t = heads[name]
            self.assertEqual([e for e, *_ in blocks], list(range(len(blocks))))
            self.assertEqual({nb for _, nb, _, _ in blocks}, {t.nbytes // len(blocks)})
            self.assertEqual(sum(nb for _, nb, _, _ in blocks), t.nbytes)
            self.assertLessEqual(blocks[0][1], role)
            self.assertEqual([o for *_, o in blocks], [t.offset + e * blocks[0][1] for e, *_ in blocks])
            self.assertEqual({p for _, _, p, _ in blocks}, {os.path.abspath(self.model)})
        # the experts' index is another file: its unaligned entries are what they were
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            expert_source.facts(self.model, info['alias'])
        self.assertEqual(f.getvalue(), "expert_types='Q4_K'\nalias_unaligned=48\n")

    def test_a_drive_home_without_a_block_split_is_refused(self):
        for homes, why in (('blk.0.attn_odd.weight\tdrive\n', 'has no split into equal blocks'),
                           ('blk.0.attn_norm.weight\tdrive\n', 'has no split into equal blocks'),
                           ('blk.0.ffn_down_exps.weight\tdrive\n', 'is not a dense weight'),
                           ('blk.0.attn_q.weight\tgpu\n', 'not a tensor<TAB>ram|drive line'),
                           ('blk.0.attn_q.weight\tram\nblk.0.attn_q.weight\tdrive\n', 'not named before')):
            with self.assertRaises(SystemExit) as e:
                self.dense(homes)
            self.assertIn(why, str(e.exception.code))

    def test_an_index_without_the_dense_weights_is_rebuilt(self):
        self.index()
        info = json.load(open(f'{self.d}/experts.json'))
        del info['dense']
        json.dump(info, open(f'{self.d}/experts.json', 'w'))
        self.assertIsNone(expert_source.current(self.d, self.model, None))
        self.assertIn('dense', self.index())


if __name__ == '__main__':
    unittest.main()
