# DeepSeek-V4.1-Flash with Hermes Agent: setup

How to run [Hermes Agent](https://github.com/NousResearch/hermes-agent) against a local DeepSeek-V4.1-Flash
server. The server needs no proxy and Hermes needs no special settings: every Hermes feature can stay at its
default. What the server does to make that work, and the measurements behind it, are in
[the adaptation record](deepseek-v41-flash-hermes-agent-changes.md).

## Server

Build and start the server as in the [README](../README.md#deepseek-v41-flash-on-an-rtx-5090):

```bash
export MODEL=/path/to/Q8_0/DeepSeek-V4.1-Flash-Q8_0-00001-of-00010.gguf
export EXPERT_PACK=/path/to/DeepSeek-V4.1-Flash-Q8_0-MXFP4.expert-pack
./run-deepseek-v41-flash-mxfp4.sh
```

The launcher already sets what an agent client needs:

| Setting | Launcher value | Why |
| --- | --- | --- |
| `--ctx-checkpoints` | 8 | 5 MiB each, in RAM: the points a conversation falls back to when a client changes an earlier turn |
| `--cache-disk`, `--cache-disk-mib` | `PROMPT_CACHE_DIR`, `PROMPT_CACHE_MIB` | saved prompt states on disk: the shared system prompt and tools, and conversations another request displaced |
| `--chat-template-file` | `models/templates/deepseek-ai-DeepSeek-V4.1.jinja` | DSML tool calls with the V4.1 tags |
| `--chat-template-kwargs` | `{"drop_thinking": false}` | keeps the reasoning of earlier turns, as generated |

Environment variables of the launcher that matter here:

- `PROMPT_CACHE_DIR` (default `${XDG_CACHE_HOME:-$HOME/.cache}/ssd-llama/deepseek-v41-flash-mxfp4`): where
  saved prompt states go. States are about 5 MiB plus 5 KiB per token (40 MiB for Hermes' 7,000-token
  system prompt with a small toolset, 75 MiB for its default 14,000 tokens). They outlive the server and are
  found again by a server running the same model file and build; files of another build are removed when the
  server starts. `PROMPT_CACHE_DIR=` turns the cache off.
- `PROMPT_CACHE_MIB` (default 16384): size limit of the directory; the least recently used states go first.
- `LLAMA_SERVER_HOST`, `LLAMA_SERVER_PORT` (default `127.0.0.1:18181`): set the host to `0.0.0.0` to serve
  another machine. The server has no authentication.
- `VRAM_RESERVE_MIB` (default 0): GPU memory to leave to another process that stays on the card. Without it
  the launcher refuses to start next to another CUDA process.

Starting `llama-server` without the launcher, pass the same options:

```bash
llama-server -m "$MODEL" ... \
    --chat-template-file models/templates/deepseek-ai-DeepSeek-V4.1.jinja \
    --chat-template-kwargs '{"drop_thinking": false}' \
    --ctx-checkpoints 8 --cache-disk /var/cache/ssd-llama/dsv41 --cache-disk-mib 16384
```

`--cache-disk` replaces `--cache-ram`; without either, a new conversation and every request that follows an
unrelated one (Hermes' title requests, for example) process their whole prompt again.

## Hermes Agent

Hermes Agent needs Python 3.14 (its dependencies are pinned for 3.14 only). Install it and point its model at
the server in `$HERMES_HOME/config.yaml` (`~/.hermes/config.yaml` by default):

```yaml
model:
  default: deepseek-v4.1-flash
  provider: custom
  base_url: http://127.0.0.1:18181/v1
  api_key: none
  context_length: 131072
```

- `base_url`: the server's address with `/v1`. Hermes uses `/v1/chat/completions`.
- `default`: the model name sent with each request. Hermes sends earlier reasoning back only for names that
  contain `deepseek`; the server restores it either way, so the name only labels the sessions.
- `context_length`: the server's context. Hermes compresses its history at 75% of it.

Everything else can stay at Hermes' defaults: all toolsets, title generation, background review, smart
approvals and the default reasoning effort (`medium`). Requests that are not part of the conversation, such
as title generation, are served between its turns: the server saves the conversation to the prompt cache and
loads it back for the next turn. A title request takes about 15 s on the test server; a message sent during
it waits for it.

Optional:

- `auxiliary.title_generation.enabled: false` saves the title request of each new session; sessions then have
  no generated title.
- `agent.local_stream_stale_timeout: 1800` gives Hermes' first request more time than the default 900 s; it
  carries the whole system prompt and tool schemas, which take the longest on a cold server.
- `reasoning_effort` changes the first line of the system prompt (DeepSeek's encoding puts the effort there),
  so each effort level has its own saved prefix. Keeping one level avoids processing the system prompt again
  when switching.

Connect Hermes directly to the DeepSeek service. A proxy that only forwards `/v1/responses`
cannot serve the chat-completions endpoint required by this configuration.

## What to expect

Times are from the test server of the [adaptation record](deepseek-v41-flash-hermes-agent-changes.md#timings)
(RTX 5090 with the target's 16 GB host limits), time to the first token:

- The first request with a new system prompt processes all of it: about 7,000 tokens with a small toolset,
  14,000 with Hermes' defaults (45 s and 108 s). Hermes writes the date and the working directory into its
  system prompt, so a new day or another directory starts a new prefix.
- Later requests in a session process only what is new (a tool result, the next user message): 2-9 s.
- A new session with the same system prompt starts from the saved prefix and processes only its first user
  message: 3.5 s, and 5.5 s right after the server restarts, when its expert caches are still empty.
- A task of a few tool calls takes 1-2 minutes when the system prompt is already saved, most of it generating
  the reasoning and answers at about 6.4 tokens/s, and about 3 minutes with a new 14,000-token one.
- Every finished response reports `timings.prompt_n` (tokens processed) and `timings.cache_n` (tokens
  reused).

The server log shows which path a request took:

| Log line | Meaning |
| --- | --- |
| `keeping the shared prompt prefix of N tokens` | the system prompt and tools were saved for later conversations |
| `found a better prompt: reuses N tokens` | the request started from a state loaded from the prompt cache |
| `restored context checkpoint (... n_tokens = N ...)` | the request fell back to a checkpoint of the current conversation |
| `forcing full prompt re-processing` | nothing reusable: the whole prompt was processed |

## Troubleshooting

- Every request processes the whole conversation: the server is older than the Hermes adaptation, or was
  started without checkpoints and prompt cache. Check the startup log for `context checkpoints are taken after
  each prompt batch` and `prompt cache is on disk`.
- A request fails right after one that produced a single token: fixed with the adaptation; update the server.
- Disk space: delete `PROMPT_CACHE_DIR` while the server is stopped to start over.
