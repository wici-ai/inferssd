#!/usr/bin/env python3
"""Extract the production parser without CUDA, retaining exact function bodies."""
from pathlib import Path
import hashlib
import json
import sys

root = Path(__file__).resolve().parents[2]
source = root / 'ggml/src/ggml-cuda/moe_stream_batch_common.cu'
text = source.read_text()
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

def block(start, end):
    a = text.index(start)
    b = text.index(end, a)
    return text[a:b].rstrip() + '\n'

names = [
    ('entry-cmp', 'static int expert_pack_entry_cmp(', 'static int expert_pack_v2_entry_cmp('),
    ('env-bool', 'static bool expert_pack_env_bool(', 'static bool common_object_cqe_enabled('),
    ('read-and-key', 'static bool expert_pack_read_exact(', 'static expert_pack_source * expert_pack_source_for_entry('),
    ('open-and-pack-v1', 'static bool expert_pack_open_data_source(', 'static bool expert_pack_v2_full_cover_down_iouring_enabled();'),
    ('alias-tsv', 'static void expert_pack_split_tsv_line(', 'static bool expert_pack_open_alias_source('),
]
fragments = [(name, block(start, end)) for name, start, end in names]
# The declarations become member functions. Every body remains byte-identical.
original = '\n'.join(value for _, value in fragments)
members = '\n'.join(line[7:] if line.startswith('static ') else line for line in original.splitlines()) + '\n'
(out / 'original-parsers.inc').write_text(original)
(out / 'member-parsers.inc').write_text(members)
types = block('struct expert_pack_entry {', 'struct expert_pack_adjacent_op_stats {')
(out / 'types.inc').write_text(types)
config_helpers = block('static bool expert_pack_env_bool(', 'static bool common_object_cqe_enabled(') + '\n' + block(
    'static void expert_pack_append_env_list(', 'static void expert_pack_v2_init_once(')
(out / 'config-helpers.inc').write_text(config_helpers)
# Keep the original sort and replacement loop. The caller supplies the same
# replace flag; the initializer's unrelated getenv and transport work stays out.
a = text.index('    std::sort(entries.begin(), entries.end(),', text.index('static void expert_pack_init_once()'))
b = text.index('    const bool replace_duplicates =', a)
c = text.index('    uint64_t replaced_duplicates = 0;', b)
d = text.index('    g_expert_pack.file =', c)
finalize = text[a:b] + text[c:d]
(out / 'finalize.inc').write_text(finalize)
manifest = {'source': str(source.relative_to(root)), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'fragments': {name: hashlib.sha256(value.encode()).hexdigest() for name, value in fragments},
            'config_helpers_sha256': hashlib.sha256(config_helpers.encode()).hexdigest(),
            'member_transform': 'remove only leading static on free-function declaration lines',
            'finalize_transform': 'retain sort and replacement loop; pass replace_duplicates as argument'}
(out / 'extraction.json').write_text(json.dumps(manifest, indent=2) + '\n')
