<!-- SPDX-License-Identifier: Apache-2.0 -->
# DeepSeek-V4.1 Engram layouts and GSQ-RCO GGUFs

[Back to documentation](README.md)

DeepSeek-V4.1 adds an Engram before some of its layers: each token hashes its
last few tokens into rows of a large embedding table. Published GGUFs store the
Engram in three layouts, and the engine loads all of them:

| Layout | Published as | Metadata | Tables |
| --- | --- | --- | --- |
| `engram.*` | mrfakename's Q8_0 GGUFs | `engram.pad_id` is a compressed token ID | one per Engram layer, with `engram_q` and `engram_k` |
| `ple.*` | unsloth's and ISTA-DASLab's GGUFs | `engram.pad_token_id` is a vocabulary token ID | one `per_layer_token_embd.weight` for every layer, with a fused `engram_gate` |
| `e4m3_e8m0_32_row264` | antirez's ds4 GGUFs | U32 arrays, offsets from the bucket sizes | one per Engram layer, 264-byte rows decoded on the CPU |

A small reader per layout maps its metadata onto one set of parameters: the
multipliers, bucket sizes and offsets, the token map and the pad as a
compressed token ID. One hash, one gate and one graph then run for every
layout, as upstream llama.cpp runs its `ple.*` models (revision 9581154a):

- A shift of the n-gram hash stops, and so does every longer one, at the start
  of the sequence and at a token the token map marks `-1`; a stopped shift
  hashes the pad.
- The gate is `sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot))`, so a dot
  product of exactly zero gates 0.50025.
- A batch of embeddings, which has no token IDs, skips the Engram.
- The shared table must cover the last bucket; a table with more rows is used
  as the file gives it.

With mmap, the default, every Engram table stays an ordinary CPU mapping of its
file, whose rows are read as tokens use them, with random-access advice, in
every layout.

## A table kept apart from the model

ISTA-DASLab publishes
[DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)
as five numbered backbone shards per quantization (`…-00001-of-00006.gguf` to
`…-00005-of-00006.gguf`) and one table, `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`,
shared by all of them. The table holds only `per_layer_token_embd.weight`; its
header carries the split numbering of the sixth shard and no architecture.

Give the table as an extra shard with `--ple` (an alias of `--shard`, as in
upstream llama.cpp). It may have any name and live in any directory or on
another drive:

```sh
inferssd serve -m /models/ISTA-IQ2_XXS/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf \
  --ple /tables/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf
```

`inferssd serve`, `inferssd cli` and `inferssd launch` pass the values to the
engine as given, for the model's facts and for the server. The engine takes them
the way upstream llama.cpp does:

- A file that holds only `per_layer_token_embd.weight` stands for the numbered
  shard that holds only the table, which is then skipped, present or missing.
  Exactly one numbered shard may be skipped, and the backbone's tensors plus the
  table must make up the tensor count its first shard declares.
- The same file replaces a table the model holds in one of its shards.
- A given file named as one of the numbered shards replaces that shard, so the
  shards can lie in different directories.
- Any other file adds its tensors to the model. An extra's split numbering is
  ignored. A tensor found twice is refused, except the table an extra replaces.

Values are comma separated, and the options can be repeated; the server also
reads `LLAMA_ARG_SHARD`. Without the table, the engine names the shard it cannot
open:

```text
error: …-00001-of-00006.gguf: llama_model_loader: failed to load GGUF split from …-00006-of-00006.gguf
```

`inferssd-server --ssd-model-facts MODEL --ple TABLE` lists the table among the
model's files. For the IQ2_XXS backbone it lists six files and 120 routed-expert
tensors: 40 layers of 384 experts, 46,080 expert roles in all, whose up, gate
and down mix IQ1_M, IQ2_XXS and IQ2_XS in eight formats.

## Tests

- `test-dsv41-engram` holds the hash to a transcription of upstream's
  `engram_hash` over random metadata, token maps with `-1` entries, tokens outside
  the map, sequence starts and embeddings, and the gate to `copysign`. Given a
  model's files (`test-dsv41-engram MODEL [SHARD…]`), it checks the reader's
  parameters and the rows of a prompt against upstream's hash over that file's
  own metadata.
- `test-dsv41-engram-layouts` (`test-llama-archs --engram-layouts`) gives one
  model in the `engram.*` and `ple.*` layouts the same logits over a prompt with
  `-1` tokens, where the `ple.*` pad is a different vocabulary token with the
  same compressed ID; reads one Engram in all three layouts into the same
  parameters and rows; and loads the model from files laid out the ways above,
  each load giving the same logits, and refuses the inputs upstream refuses.
- `test-ple-args` covers the options and `LLAMA_ARG_SHARD`, and
  `ssd_model_facts` the facts of a split model whose table is given apart from it.
- `dsv41_mixed_iq_cpu_sink_decode` checks the CPU expert sink on mixed IQ2_XXS,
  IQ2_XS, IQ2_S and IQ3_XXS experts against the ordinary graph.
  `test-ssd-vram-mm --dsv41-iq` and `test-ssd-vram-generic --dsv41-iq` check the
  CUDA expert tier on the same formats.
