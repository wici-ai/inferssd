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

    prefix-cache 1|<size>:<header hash>,...|<model description>|<build>

with the size and an FNV-1a hash of the header (the metadata and the tensor list) of each file the model is loaded
from, the other files of a split model included, and the build that saved it. A server takes a file of its own prompt
cache only from its own build, and a prefix cache from any build: it compares the fingerprint up to the build. Where
the files are and when they were written are not in it. A
state saved with another layout (another KV type, or flash attention on where it was off) fails to restore; the
server then leaves that file and processes the prefix itself.

## The server

`--prefix-cache DIR` names the directory of a model's prefix caches. The server reads the `*.prompt` files there
whose fingerprint is the model's, and looks for a state among them as among its own cached prompts. It never writes,
removes or evicts them, and they do not count against `--cache-disk-mib`. It reads the directory again when the
directory changes, so a file that arrives while the server runs serves the next request.

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

For a new engine, each published prefix cache is made again. If the new state differs from the published one by no
more than two backends differ, the file is kept and the new engine is added to its `engines`; otherwise the new file
is published for it. A state computed twice on one GPU is not the same bytes either, so the comparison is by size of
difference, not equality.
