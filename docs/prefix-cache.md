# Prefix caches

A coding agent starts every conversation with the same context: its system prompt and its tools, 10k to 50k tokens.
The server keeps the state after that shared prefix in its prompt cache, so only the first request on a machine
waits for it. A prefix cache is that state computed once, elsewhere, for a model, and downloaded with it: the first
request of Claude Code or Codex then starts after the prefix too.

## What makes a prefix cache usable

A prefix cache is a cached prompt file (`*.prompt`, the format of `--cache-disk`) whose tokens are the shared prefix
of an app's requests as the server renders them. The server uses one only when a request starts with exactly those
tokens, so a prefix cache made for another version of an app, another system or another chat template is never
restored; it is just not used.

The state in it is what the model computed for those tokens. Another backend, another machine or another build
computes the same values up to rounding, as a GPU and a CPU do, and the server takes those differences as it takes
them between its own prompt passes. A prefix cache is a file of a server's prompt cache on disk, whose fingerprint
is

    prefix-cache 1|[settings=<hash>|]<size>:<header hash>,...|<model description>|<build>|settings-cache=1|adapter-cache=1

with the size and an FNV-1a hash of the header (the metadata and the tensor list) of each file the model is loaded
from, the other files of a split model included, and the build that saved it. A server takes a file of its own prompt
cache only from its own build, and a prefix cache from any build: it compares the fingerprint up to the build. Where
the files are and when they were written are not in it. A state saved with another layout (another KV type, or flash
attention on where it was off) fails to restore; the server then leaves that file and processes the prefix itself.

The `settings=<hash>|` field names what else a state's values depend on, so that its states serve only servers that
compute the same: GGUF metadata overrides, RoPE or YaRN settings, an attention type, control vectors, a quantized KV
cache stored without its rotation by `LLAMA_ATTN_ROT_DISABLE`, and for a model with long and short RoPE factors
(LongRoPE) the factors the server's context selects, the long ones when a sequence's context is past the model's
original one. A server that runs the model as its files describe it, without LongRoPE factors, has no field. The
publisher refuses a file with the field, and so takes no prefix cache of a LongRoPE model.

`settings-cache=1` says that the writer named those settings, and is in every file of a server that does. A file
without it, of a server before it, says nothing of the settings that computed it, which may have been others: it
serves as a prefix cache only when it is one of the published files, which are of the model's own settings: those
that `tools/server/inferssd/prompt-cache-legacy.inc` names by size and SHA-256, every file the published list named
before the marker. Reading the directory reads only the headers, where the size of a published file is enough; the
first request that would restore such a file hashes it, once for each version of it (about 2 seconds for a 1 GB file
on an Apple M5), and a file that is not a published one goes, the request processing the prefix itself. A writable
prompt cache takes only files of its own build, and so drops those of a server before it.

Writable prompt caches keep base-model and ordinary LoRA states separate. Reuse with active LoRA requires
matching adapter contents and exact scales. Active aLoRA states stay local to their slot, where activation
position governs reuse. Older writable files without adapter provenance are rebuilt on the first launch
with the new cache fingerprint. New published prefixes must carry the fingerprint of a server that records its
settings and adapters (ending in `|settings-cache=1|adapter-cache=1`). Previously published prefixes remain readable
after verification against their published sizes and SHA-256 digests.

Explicit slot saves also record the model, build, and effective adapter key. Restoring a compatible
save keeps its state available to automatic or pinned slot selection. A request with a different
adapter recomputes the prompt. The server rejects older explicit slot files without provenance.

## The server

`--prefix-cache DIR` names the directory of a model's prefix caches. The server reads the `*.prompt` files there
whose fingerprint is the model's, and looks for a state among them as among its own cached prompts. It never writes,
removes or evicts them, and they do not count against `--cache-disk-mib`. It reads the directory again when the
directory changes, so a file that arrives while the server runs serves the next request.

A server that wakes from sleep loads the model again and derives its cache from that load, as a server that starts
does: its fingerprint, its settings, its attention mode and its checkpoints, and a prompt cache of that fingerprint.

A server whose attention is not causal (`--attention non-causal`, or a model or metadata override that turns causal
attention off) takes no prefix cache and keeps no prompt cache or context checkpoints: each row of its state depends on
every token of the batch that computed it, the tokens after the row too, so the state serves no other request. Its
slots evaluate every prompt in full, in the ids it was sent in.

A context shift (`--context-shift`) moves the rows after the tokens it discards to their places, and those rows still
depend on the discarded tokens; so do the chunks `--cache-reuse` moves into place. The server saves no state with
moved rows, in its prompt cache or as a prefix, drops the slot's checkpoints at a shift, and a later prompt on the slot
reuses none of the moved rows, nor takes their ids for its text. An explicit slot save of such a state is restored
without reuse.

`inferssd serve` passes `<cache directory>/prefix-cache/<model>`, beside the prompt caches
(`<cache directory>/prompts/<model>`), `<model>` being the model's first file name without the shard suffix and
`.gguf`. The cache directory is `~/Library/Caches/inferssd` on macOS and `%LOCALAPPDATA%\inferssd` on Windows.

## The list

The published prefix caches are listed at `https://appapi.wici.ai/versioning/inferssd/config/prefix-cache.json`:

```json
{
  "schema": "inferssd-prefix-cache 1",
  "files": [
    {
      "engines": ["<40-hex engine commit>", "..."],
      "os": "windows",
      "model": "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
      "app": "claude-code",
      "app_version": "2.1.328",
      "tokens": 10592,
      "size": 283085613,
      "sha256": "<64-hex>",
      "url": "https://cdn.wici.ai/apps/inferssd/prefix-cache-<sha256>.prompt"
    }
  ]
}
```

- `engines` are the engine commits (`inferssd --version --json`, `revision`) the file is published for.
- `os` is the system the app ran on: an app's prompt names it, in its paths and its tools.
- `model` is the model's first GGUF file, the one `inferssd serve -m` takes.
- Files are named by their SHA-256 and never change.

The WiCi AI desktop app keeps each local model's prefix-cache directory to the files the list names for the installed engine and
the system: it downloads them right after the model, checks the list again every hour, and removes the files the list
no longer names and the directories of models that are gone. While the list names the installed engine nowhere, as
for an engine published within the last hour, it leaves the directory as it is. Offline it changes nothing.

## Making them

A prefix cache is made by running the app's request through a server of the published engine with an empty prompt
cache: the server saves the state after the shared prefix. The request is the one the app sends on that system with
the configuration the desktop app gives it, with the user's identifiers already replaced by the server's stand-ins
(`tools/server/inferssd/identifiers.h`), so the tokens are the ones every user's server renders.

The stand-ins are those of the system the server runs on, so a server renders another system's requests as that
system's servers do only when told which system they come from. With `INFERSSD_IDENTIFIERS_OS` set to `macos`,
`windows` or `linux`, the server uses that system's stand-ins and spellings for the home directory in `HOME`
(`USERPROFILE` for `windows`) and, for `macos`, the temporary directory in `TMPDIR`, and that system's text where an
app version describes a tool differently on each system (`tools/server/inferssd/app-versions.h`): set them to the
folders the app ran with, and a faster machine of any system makes the prefix caches of all of them. Only the state's
numbers then differ, as between two backends.

For a new engine, each published prefix cache is made again. If the new state differs from the published one by no
more than two backends differ, the file is kept and the new engine is added to its `engines`; otherwise the new file
is published for it. A state computed twice on one GPU is not the same bytes either, so the comparison is by size of
difference, not equality.

The publisher updates the entry for the requested system, model, app, app version, and file SHA-256. A new file must
end in `|settings-cache=1|adapter-cache=1`. It can add an engine to an existing entry of a file without that when the
size and SHA-256 match both that entry and the native import inventory in the named engine commit. Fetch the engine
commits before publishing so the publisher can inspect their source.
The publisher keeps the existing download and verifies the bytes it serves before writing the updated list.
